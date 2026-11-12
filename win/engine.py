"""모션 추출 엔진 (ComfyUI 노드가 win/mocap_cli.py 를 통해 별도 프로세스로 실행).

    eng = MocapEngine()
    res = eng.run("dance.mp4", target="Beatrice", out_dir="D:/out")
    res -> {"bvh", "mesh_glb", "skeleton_glb", "preview_mp4", "frames", "work_dir"}
"""
import glob
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wincompat  # noqa: E402,F401

import torch  # noqa: E402

from win import model_store, motion_fix, rig_prep  # noqa: E402

CONFIG = os.path.join(REPO, "win", "mca_inference.yaml")
MAX_FRAMES = 301          # 원본 추론 코드의 시퀀스 상한 (30fps 기준 약 10초)


def _species_map(base):
    """캐릭터 이름 → 참조 시퀀스 ('이름#이름-000Rest/y0'). 정렬상 첫 시퀀스 = 기준(기본) 포즈."""
    pose_root, train_root = os.path.join(base, "bvh_pose"), os.path.join(base, "npz_train_image_only")
    out = {}
    if not os.path.isdir(pose_root):
        return out
    for d in sorted(os.listdir(pose_root)):
        sp = d.split("#")[0]
        if sp in out:
            continue
        if (os.path.exists(os.path.join(pose_root, d, "y0.npz"))
                and os.path.exists(os.path.join(train_root, d, "y0.npz"))):
            out[sp] = f"{d}/y0"
    return out


def list_targets():
    return sorted(_species_map(rig_prep.ZOO))


def to_30fps(src, dst, start=0.0, seconds=10.0):
    """입력 영상을 30fps 로 다시 인코딩 (모델은 30fps 프레임으로 학습, BVH 도 1/30초 간격).
    start 초부터 seconds 초만 사용하고 오디오는 버림."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", f"{float(start):.3f}"]
    cmd += ["-i", src]
    if seconds and seconds > 0:
        cmd += ["-t", f"{float(seconds):.3f}"]
    cmd += ["-an", "-vf", "fps=30", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "fast", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"Video conversion failed: {r.stderr[-500:]}")
    return dst


class MocapEngine:
    def __init__(self, log=print):
        if not torch.cuda.is_available() and os.environ.get("MCA_ALLOW_CPU") != "1":  # MCA_ALLOW_CPU: 개발 테스트용
            raise RuntimeError("No CUDA GPU found. MoCapAnything needs an NVIDIA GPU.")
        from utils.config_utils import load_yaml_config, instantiate_from_config
        from inference.video2pose2rot import DinoPipe
        from preprocess.briarmbg import BriaRMBG   # 이름만 원본, 내부는 BiRefNet(MIT)

        self.log = log
        self.load_yaml_config = load_yaml_config
        cfg0 = load_yaml_config(CONFIG)
        model_store.ensure(["mocap", "dinov2", "birefnet"], log)   # 없으면 처음 한 번 자동 다운로드
        ck = model_store.mocap_ckpt()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log("[engine] loading models")
        self.rmbg = BriaRMBG.from_pretrained("mit").eval().to(self.device)
        self.pipe = DinoPipe(self.device)
        self.model = instantiate_from_config(cfg0["model"]).float().eval()
        sd = torch.load(ck, map_location="cpu")
        sd = next((sd[k] for k in ("model_state", "model", "model_state_dict", "state_dict")
                   if isinstance(sd, dict) and k in sd), sd)
        self.model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()})
        self.model.to(self.device)
        log("[engine] models ready")

    def run(self, video_path, target, out_file, fmt="fbx", work_root=None, use_sam=True, start=0.0, seconds=10.0,
            keep_work=False, stabilize="fix flips", mirror_side="off", mirror_timing="match original",
            foot_lock=False, in_place=False):
        """Capture motion and save ONE file (out_file, fmt = fbx | glb | bvh).
        Preview GLBs for ComfyUI's Preview 3D are written to the work folder (ComfyUI temp), not the output.
        stabilize / mirror_side / mirror_timing / foot_lock / in_place: clean-up passes
        (win/motion_fix.py)."""
        from inference.video2pose2rot import video_to_frames, inference

        sp_map = _species_map(rig_prep.ZOO)
        if target not in sp_map:
            raise RuntimeError(f"'{target}' is not a prepared character. Prepared: {sorted(sp_map)}")
        fmt = fmt.lower()
        char_fbx = os.path.join(rig_prep.WORKSPACE, "fbx_in", target, f"{target}.fbx")
        if fmt in ("fbx", "glb") and not os.path.isfile(char_fbx):
            raise RuntimeError(f"Character file not found: {char_fbx}. Prepare the character again (overwrite on).")
        base = rig_prep.ZOO
        char_root = os.path.join(base, "characters_face_zplus")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        tmp = os.path.join(work_root or os.path.join(os.path.dirname(out_file), "_mocap_work"), f"{target}_{stamp}")
        seq_name = f"{target}#req{stamp}"
        frames_dir = os.path.join(tmp, "frames", seq_name)
        os.makedirs(frames_dir, exist_ok=True)

        seconds = min(float(seconds or 10.0), MAX_FRAMES / 30.0)
        vid30 = to_30fps(video_path, os.path.join(tmp, "input_30fps.mp4"), start, seconds)

        n = 0
        if use_sam:
            try:
                n = self._sam_frames(vid30, tmp, frames_dir)
                self.log(f"[engine] SAM2 isolated the subject: {n} frames" if n else "[engine] SAM2 found no subject")
            except Exception as e:  # noqa: BLE001
                self.log(f"[engine] SAM2 failed, falling back to background removal: {e}")
                n = 0
        if n == 0:
            for f in glob.glob(os.path.join(frames_dir, "*")):
                os.remove(f)
            n = video_to_frames(vid30, frames_dir)
        if n == 0:
            raise RuntimeError("Could not read any frames from the video.")
        self.log(f"[engine] {n} frames -> running inference (character: {target})")

        cfg = self.load_yaml_config(CONFIG)
        cfg["data"].update(base_dir=base,
                           scale_dict_path=os.path.join(base, "cache", "__mesh2pose1002_species_scale_cache.pkl"),
                           character_dir=char_root,
                           bvh_roots=[os.path.join(base, "bvh")],
                           image_roots=[os.path.join(tmp, "frames")], wild_flag=True)
        cfg["data"]["retarget"].update(ref_seq=sp_map[target], ref_idx=0)
        cfg["output"].update(save_dir=os.path.join(tmp, "out"), output_tag="mca", blender_path=None, fps=30,
                             make_video=False)
        cfg["export_glb"] = True
        inference(cfg=cfg, device=self.device, attention_design=cfg["model"]["attention_kwargs"],
                  model=self.model, pipe=self.pipe, rmbg_net=self.rmbg, seq_name=seq_name,
                  image_folder=frames_dir, stage_cb=lambda nm: self.log(f"[engine] stage: {nm}"))

        out_root = os.path.join(tmp, "out")

        def found(pat):
            return next(iter(sorted(glob.glob(os.path.join(out_root, "**", pat), recursive=True))), None)

        bvh = found("*_rot6d_pred.bvh")
        if not bvh:
            raise RuntimeError("Inference produced no BVH. See the console log.")
        mesh_glb, skel_glb = found("*_mesh.glb"), found("*_skeleton.glb")

        # clean-up passes; the previews are rebuilt from the cleaned motion so they show the saved result
        fixed = os.path.join(tmp, "motion_fixed.bvh")
        if motion_fix.process(bvh, fixed, stabilize, mirror_side, mirror_timing, foot_lock, in_place,
                              log=self.log):
            bvh = fixed
            try:
                from utils.glb_export import export_skinned_glb
                m2, s2 = os.path.join(tmp, "mesh_fixed.glb"), os.path.join(tmp, "skeleton_fixed.glb")
                export_skinned_glb(bvh, os.path.join(char_root, target), m2, s2, fps=30, validate=False)
                mesh_glb, skel_glb = m2, s2
            except Exception as e:  # noqa: BLE001
                self.log(f"[engine] preview of the cleaned motion failed (the saved file is not affected): {e}")
        if not (mesh_glb and skel_glb):
            from utils.glb_export import export_skinned_glb
            mesh_glb, skel_glb = os.path.join(tmp, "mesh.glb"), os.path.join(tmp, "skeleton.glb")
            export_skinned_glb(bvh, os.path.join(char_root, target), mesh_glb, skel_glb, fps=30, validate=False)

        # preview files for ComfyUI (kept in temp; no '#' in the names)
        prev_dir = os.path.join(os.path.dirname(tmp), "_preview")
        os.makedirs(prev_dir, exist_ok=True)
        res = {"frames": n, "work_dir": tmp}
        for key, src, nm in (("mesh_glb", mesh_glb, "mesh"), ("skeleton_glb", skel_glb, "skeleton")):
            if src and os.path.exists(src):
                dst = os.path.join(prev_dir, f"{target}_{stamp}_{nm}.glb")
                shutil.copy2(src, dst)
                res[key] = dst

        # the one saved file
        self.log(f"[engine] stage: export {fmt.upper()}")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if fmt == "bvh":
            shutil.copy2(bvh, out_file)
        else:
            blender = rig_prep.find_blender(rig_prep.saved_blender())
            if not blender:
                raise RuntimeError("Blender not found (needed to write FBX/GLB). Choose BVH, or set blender_path "
                                   "in MCA Prepare Character and prepare again.")
            cmd = [blender, "-b", *wincompat.blender_args(), "--python",
                   os.path.join(REPO, "win", "blender_export_anim.py"), "--",
                   "--char", char_fbx, "--bvh", bvh, "--out", out_file, "--format", fmt, "--fps", "30"]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            for ln in (r.stdout or "").splitlines():
                if ln.startswith("[export]"):
                    self.log(f"[engine] {ln}")
            if r.returncode != 0 or not os.path.isfile(out_file):
                tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-15:]
                raise RuntimeError(f"{fmt.upper()} export failed (Blender):\n" + "\n".join(tail))
        res["file"] = out_file
        if not keep_work:
            shutil.rmtree(tmp, ignore_errors=True)
        self.log(f"[engine] done -> {out_file}")
        return res

    def _sam_frames(self, video_path, tmp, frames_dir):
        """SAM2 로 영상의 주 인물(전경 마스크와 가장 많이 겹치는 후보)을 분리해 RGBA 프레임으로 저장."""
        from win import sam_utils as su   # SAM2 (transformers 판, 원본 demo/sam_utils.py)
        from inference.video2pose2rot import video_to_frames
        from PIL import Image

        model_store.ensure(["sam2"], self.log)
        if not su.available():
            return 0
        orig, sam = os.path.join(tmp, "orig"), os.path.join(tmp, "sam")
        os.makedirs(orig, exist_ok=True)
        os.makedirs(sam, exist_ok=True)
        video_to_frames(video_path, orig)
        f0 = sorted(glob.glob(os.path.join(orig, "*.png")))
        if not f0:
            return 0
        for f in f0:
            Image.open(f).convert("RGB").save(os.path.join(sam, os.path.basename(f)[:-4] + ".jpg"))
        seed = len(f0) // 2           # 중간 프레임에서 시작 (첫 프레임의 타이틀/검은 화면 회피)
        _, segs = su.candidate_layers(f0[seed], os.path.join(tmp, "cand"), topk=4, prior=self._fg_prior(f0[seed]))
        if not segs:
            return 0
        return su.track_to_rgba(sam, segs[0], orig, frames_dir, seed_idx=seed)

    def _fg_prior(self, img_path):
        """배경 제거 모델 전경 마스크 — SAM 후보 정렬용 (원본 앱의 _fg_prior 와 같은 처리)."""
        try:
            import cv2
            import torchvision.transforms.functional as TF

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            t = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1) / 255.0
            t = torch.nn.functional.interpolate(t[None], size=(1024, 1024), mode="bilinear", antialias=True)[0]
            inp = TF.normalize(t, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])[None]
            with torch.no_grad():
                r = self.rmbg(inp)[0][0]
            m = torch.nn.functional.interpolate(r, size=(h, w), mode="bilinear")[0, 0]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            return (m > 0.5).cpu().numpy()
        except Exception as e:  # noqa: BLE001
            self.log(f"[engine] foreground prior failed (ignored): {e}")
            return None
