"""필요한 모델 확인 + 처음 한 번 자동 다운로드 (다른 ComfyUI 커스텀 노드들처럼).

저장 위치: ComfyUI/models/mocapanything/
  checkpoints/  MoCapAnything V2 가중치
  hf/           DINOv2, BiRefNet, SAM2, T5 (HuggingFace 캐시 형식)

 키        저장소                              라이선스     크기(약)  쓰는 곳
 mocap     kehong/MoCapAnythingV2-weights      MIT          0.46GB   모션 추출
 dinov2    facebook/dinov2-large               Apache-2.0   1.2GB    캐릭터 준비, 모션 추출
 birefnet  ZhengPeng7/BiRefNet                 MIT          0.9GB    캐릭터 준비, 모션 추출 (배경 제거)
 sam2      facebook/sam2.1-hiera-base-plus     Apache-2.0   0.3GB    모션 추출 (use_sam 켰을 때)
 t5        google-t5/t5-base                   Apache-2.0   0.9GB    캐릭터 준비 (본 이름 임베딩)
           (옛 이름 t5-base 로 이미 받아 둔 PC는 그것을 그대로 씀)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wincompat  # noqa: E402

# 필요한 파일만 받기 (같은 가중치의 .bin / .pt / tf / flax 사본은 건너뜀)
_HF_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.model", "*.txt"]

MODELS = {
    "mocap": ("kehong/MoCapAnythingV2-weights", 0.46, ["*.pt", "*.md"]),
    "dinov2": ("facebook/dinov2-large", 1.2, _HF_PATTERNS),
    "birefnet": ("ZhengPeng7/BiRefNet", 0.9, _HF_PATTERNS),
    "sam2": ("facebook/sam2.1-hiera-base-plus", 0.3, _HF_PATTERNS),
    "t5": ("google-t5/t5-base", 0.9, _HF_PATTERNS),   # 정식 이름 (옛 이름 t5-base 는 Xet 다운로드에서 404)
}
CKPT_NAME = "video2pos2rot_epoch60.pt"


def mocap_ckpt():
    return os.path.join(wincompat.CKPT_DIR, CKPT_NAME)


def _marker(key):
    return os.path.join(wincompat.HF_DIR, f".complete-{key}")


def _snapshot(repo, pats, **kw):
    from huggingface_hub import snapshot_download

    if os.name == "nt":
        # huggingface_hub downloads files in 8 threads. On Windows without Developer Mode, a worker can call
        # os.symlink before the one-time "are symlinks supported?" check has finished in another thread
        # -> OSError [WinError 1314]. One worker makes the check run first, then files are copied instead.
        kw.setdefault("max_workers", 1)
    return snapshot_download(repo, allow_patterns=pats, **kw)


def t5_repo():
    """T5 저장소 이름. 옛 이름(t5-base)으로 이미 다 받아 둔 PC는 다시 받지 않고 그것을 쓰고, 아니면 정식 이름."""
    try:
        with open(_marker("t5"), encoding="utf-8") as f:
            got = f.read().strip()
        if got:
            return got
    except OSError:
        pass
    return MODELS["t5"][0]


def ensure(keys, log=print):
    """Download the models in keys if missing. When a model is complete, no network access is made."""
    for key in keys:
        repo, gb, pats = MODELS[key]
        if key == "mocap":
            if os.path.exists(mocap_ckpt()):
                continue
            log(f"[models] downloading {repo} (~{gb} GB, first run only) -> {wincompat.CKPT_DIR}")
            _snapshot(repo, pats, local_dir=wincompat.CKPT_DIR)
            continue
        # A marker file is written only after a download finished, so an interrupted download
        # (its snapshot folder already exists) is resumed instead of being treated as complete.
        if os.path.exists(_marker(key)):
            continue
        log(f"[models] downloading {repo} (~{gb} GB, first run only) -> {wincompat.HF_DIR}")
        _snapshot(repo, pats)
        os.makedirs(wincompat.HF_DIR, exist_ok=True)
        with open(_marker(key), "w", encoding="utf-8") as f:
            f.write(repo + "\n")
    return True
