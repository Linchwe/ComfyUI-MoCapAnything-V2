"""RMBG-1.4(비상업 라이선스) 대체 배경 제거 모듈 — MIT 라이선스 모델만 사용.

원본 코드는 `BriaRMBG.from_pretrained("briaai/RMBG-1.4")` 로 만든 네트워크를
`net(x)[0][0]` 형태로 호출합니다.
 - 입력 x : [1, 3, H, W], (이미지 - 0.5) 로 정규화된 값 (0~1 이미지 기준)
 - 출력   : ([mask, ...], [...]) 이고 mask 는 [1, 1, H, W] 0~1 확률

이 모듈의 MitMatting 은 같은 입출력을 지키면서 내부만 바꿉니다.
 1순위 BiRefNet (ZhengPeng7/BiRefNet, MIT)            — transformers 원격 코드로 로드
 2순위 InSPyReNet (transparent-background 패키지, MIT) — ComfyUI 환경에 이미 설치된 경우가 많음

환경변수 MCA_MATTING = auto | birefnet | inspyrenet 로 강제할 수 있습니다.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BIREFNET_ID = os.environ.get("MCA_BIREFNET_ID", "ZhengPeng7/BiRefNet")
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load_birefnet():
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(BIREFNET_ID, trust_remote_code=True)
    return model.eval()


def _load_inspyrenet():
    from transparent_background import Remover

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return Remover(mode="base", jit=False, device=dev)


class MitMatting(nn.Module):
    """BriaRMBG 와 같은 방식으로 호출되는 MIT 라이선스 배경 제거기."""

    def __init__(self, backend, model):
        super().__init__()
        self.backend = backend
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self._remover = model
        # parameters() 로 장치를 읽는 코드(win/engine.py 의 _fg_prior 등)를 위해 더미 파라미터 하나
        self._device_probe = nn.Parameter(torch.zeros(1), requires_grad=False)

    # 원본과 같은 이름의 생성자. 인자로 무엇이 오든(RMBG-1.4 경로 등) 무시합니다.
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        choice = os.environ.get("MCA_MATTING", "auto").lower()
        errors = []
        if choice in ("auto", "birefnet"):
            try:
                print(f"[matting] loading BiRefNet ({BIREFNET_ID}, MIT)")
                return cls("birefnet", _load_birefnet())
            except Exception as e:  # 원격 코드/버전 문제 등
                errors.append(f"BiRefNet: {e}")
                print(f"[matting] BiRefNet failed to load: {e}")
        if choice in ("auto", "inspyrenet"):
            try:
                print("[matting] loading InSPyReNet (transparent-background, MIT)")
                return cls("inspyrenet", _load_inspyrenet())
            except Exception as e:
                errors.append(f"InSPyReNet: {e}")
                print(f"[matting] InSPyReNet failed to load: {e}")
        raise RuntimeError("Could not load any background-removal model:\n  " + "\n  ".join(errors))

    @property
    def device(self):
        return self._device_probe.device

    @torch.no_grad()
    def forward(self, x):
        img = (x.float() + 0.5).clamp(0.0, 1.0)          # 원본 정규화 되돌리기 → 0~1 RGB
        h, w = img.shape[-2:]
        if self.backend == "birefnet":
            dev = self._device_probe.device
            dtype = next(self.model.parameters()).dtype
            inp = F.interpolate(img, size=(1024, 1024), mode="bilinear", align_corners=False)
            inp = (inp - _IMAGENET_MEAN.to(inp)) / _IMAGENET_STD.to(inp)
            out = self.model(inp.to(dev, dtype))
            logits = out[-1] if isinstance(out, (list, tuple)) else getattr(out, "logits", out)
            mask = logits.float().sigmoid()
        else:
            from PIL import Image

            arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            m = self._remover.process(Image.fromarray(arr), type="map")
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            mask = torch.from_numpy(m / 255.0)[None, None].to(x.device)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = mask.to(x.device)
        return [mask], [None]
