"""내 캐릭터(리그) 전처리 — 원본 examples/custom_rig/run.sh + preprocess/run_pipeline.sh 의 윈도우 버전.

캐릭터 FBX(필수)와 모션 FBX(선택)를 받아서 MoCapAnything 이 추론에 쓰는 데이터를 만듭니다.
결과는 workspace/zoo 아래에 쌓이고, ComfyUI 'MCA 모션 추출' 노드의 target 목록에 나타납니다.

CLI:
  python win/rig_prep.py --name Beatrice --base D:/char/Beatrice.fbx [--motion a.fbx --motion b.fbx]
                         [--blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"]
                         [--front +z|-z|+x|-x]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wincompat  # noqa: E402

WORKSPACE = wincompat.WORKSPACE   # 기본: ComfyUI/models/mocapanything/characters
ZOO = os.path.join(WORKSPACE, "zoo")
RUNNER = os.path.join(REPO, "win", "run_script.py")

FRONT_CHOICES = {
    "+z": [0.0, 0.0, 1.0],
    "-z": [0.0, 0.0, -1.0],
    "+x": [1.0, 0.0, 0.0],
    "-x": [-1.0, 0.0, 0.0],
}


# ----------------------------------------------------------------------------- 유틸
def safe_name(name):
    """폴더/파일 이름으로 쓸 수 있게: 영문, 숫자, - 만 남김 (# 와 공백은 파이프라인에서 의미가 있음)."""
    s = re.sub(r"[^A-Za-z0-9\-]+", "", name.replace(" ", "-").replace("_", "-"))
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "MyRig"


def saved_blender():
    """Blender used for the last character preparation (MCA Motion Capture reuses it for FBX/GLB export)."""
    try:
        with open(os.path.join(WORKSPACE, "blender_path.txt"), encoding="utf-8") as f:
            p = f.read().strip()
        return p if p and os.path.isfile(p) else None
    except OSError:
        return None


def find_blender(hint=None):
    """블렌더 실행 파일 찾기: 인자 → BLENDER_BIN/BLENDER 환경변수 → 기본 설치 경로 → PATH."""
    cands = [hint, os.environ.get("BLENDER_BIN"), os.environ.get("BLENDER")]
    if wincompat.IS_WIN:
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        found = []
        for r in roots:
            found += glob.glob(os.path.join(r, "Blender Foundation", "Blender*", "blender.exe"))
        # 버전 숫자가 큰 것 우선 (4.5 > 4.2 ...)
        found.sort(key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(os.path.dirname(p)))] or [0],
                   reverse=True)
        cands += found
        steam = r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
        cands.append(steam)
    cands.append(shutil.which("blender"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


class Log:
    """진행 로그를 콘솔·로그 파일·콜백에 동시에 흘려보냄."""

    def __init__(self, cb=None, log_path=None):
        self.cb = cb
        self.f = open(log_path, "a", encoding="utf-8") if log_path else None

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        if self.cb:
            self.cb(line)

    def close(self):
        if self.f:
            self.f.close()


def _env(blender):
    env = os.environ.copy()
    env.update(
        ZOO_ROOT="zoo",                     # 원본 스크립트 여러 곳이 'zoo' 상대경로를 하드코딩 → cwd=WORKSPACE
        BLENDER=blender,
        BLENDER_BIN=blender,
        PYTHON=sys.executable,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        MCA_WORKERS=os.environ.get("MCA_WORKERS", "2"),
    )
    return env


def _run(cmd, env, log, title):
    log(f"── {title}")
    p = subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-40:]
        # 진행바(tqdm) 잡음은 화면에서 줄임
        if "it/s]" in line or "s/it]" in line:
            continue
        log("   " + line[:300])
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{title} failed (exit code {rc})\n" + "\n".join(tail[-15:]))


def _py(script, *args):
    return [sys.executable, RUNNER, os.path.join(REPO, script), *args]


def clean_rig(name):
    """같은 이름으로 다시 준비할 때 이전 결과를 지움."""
    pats = [
        f"fbx_in/{name}", f"zoo/_fixed/{name}",
        f"zoo/characters_fix_facezplus/{name}", f"zoo/characters_face_zplus/{name}",
        f"zoo/motions/{name}#*", f"zoo/motions_face_zplus/{name}#*",
        f"zoo/bvh/{name}#*", f"zoo/bvh_pose/{name}#*", f"zoo/video/{name}#*", f"zoo/image/{name}#*",
        f"zoo/npz_train_image_only/{name}#*", f"zoo/video_fast_work/jobs/{name}#*",
        f"zoo/video_fast_work/limited_bvh/{name}#*", f"align_ref/{name}", f"zoo/ref_images/{name}.jpg",
    ]
    for pat in pats:
        for p in glob.glob(os.path.join(WORKSPACE, pat)):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)


def list_rigs():
    """준비가 끝난 내 캐릭터 이름 목록."""
    out = []
    root = os.path.join(ZOO, "characters_face_zplus")
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isdir(d) and os.path.exists(os.path.join(d, "base_mesh.obj"))
                and glob.glob(os.path.join(ZOO, "npz_train_image_only", f"{name}#*", "y0.npz"))):
            out.append(name)
    return out


# ----------------------------------------------------------------------------- 메인
def prepare_rig(name, base_fbx, motion_fbxs=(), blender=None, front="+z", wiggle=True,
                render_frames=8, log_cb=None):
    """캐릭터 하나를 전처리. 성공하면 캐릭터 이름(정리된 이름)을 반환."""
    name = safe_name(name)
    blender = find_blender(blender)
    if not blender:
        raise RuntimeError("Blender not found. Set blender_path to blender.exe (Blender 4.2+).")
    try:
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(os.path.join(WORKSPACE, "blender_path.txt"), "w", encoding="utf-8") as f:
            f.write(blender)
    except OSError:
        pass
    if not base_fbx or not os.path.isfile(base_fbx):
        raise RuntimeError(f"Character FBX not found: {base_fbx}")

    os.makedirs(os.path.join(WORKSPACE, "logs"), exist_ok=True)
    log = Log(log_cb, os.path.join(WORKSPACE, "logs", f"prepare_{name}.log"))
    t0 = time.time()
    try:
        log(f"Preparing character '{name}' - Blender: {blender}")
        from win import model_store
        model_store.ensure(["t5", "dinov2", "birefnet"], log)   # 오래 걸리는 작업 도중에 멈추지 않게 먼저 확인
        clean_rig(name)
        env = _env(blender)

        # 0) 입력 정리: workspace/fbx_in/<name>/<name>.fbx (+ 모션)
        #    원본 규칙: 폴더 안에서 이름이 캐릭터 이름과 같은 FBX = 기본(메시+뼈대), 나머지 = 모션
        src_dir = os.path.join(WORKSPACE, "fbx_in", name)
        os.makedirs(src_dir, exist_ok=True)
        # 캐릭터 FBX 는 기본 포즈 2프레임 액션을 붙여서 넣음 (원본 FBX 정리 단계가 액션 없는 FBX 를 처리 못함)
        _run([blender, "-b", *wincompat.blender_args(), "--python",
              os.path.join(REPO, "win", "blender_make_wiggle.py"), "--", "--in", base_fbx,
              "--out", os.path.join(src_dir, f"{name}.fbx"), "--mode", "rest"],
             env, log, "1/9 (a) read character FBX (Blender)")
        motions = [m for m in (motion_fbxs or []) if m]
        for i, m in enumerate(motions):
            if m and os.path.isfile(m):
                stem = safe_name(os.path.splitext(os.path.basename(m))[0])
                if "TPOSE" in stem.upper():
                    stem = stem.upper().replace("TPOSE", "TP")
                # 사용자 모션도 캐릭터와 같은 단위/본 이름 규칙으로 맞춰서 넣음 (애니메이션은 그대로)
                _run([blender, "-b", *wincompat.blender_args(), "--python",
                      os.path.join(REPO, "win", "blender_make_wiggle.py"), "--", "--in", m,
                      "--out", os.path.join(src_dir, f"{name}-M{i + 1:02d}{stem}.fbx"), "--mode", "keep"],
                     env, log, f"1/9 (c) read motion FBX {i + 1}: {os.path.basename(m)} (Blender)")
            elif m:
                raise RuntimeError(f"Motion FBX not found: {m}")
            if not os.path.isfile(os.path.join(src_dir, f"{name}-M{i + 1:02d}{stem}.fbx")):
                raise RuntimeError(f"Motion FBX has no moving animation: {m}")
        if not motions:
            # 모션 FBX 를 따로 안 주면 캐릭터 FBX 안의 애니메이션을 참조 모션으로 씀 (원본처럼 실제 동작이 참조)
            out = os.path.join(src_dir, f"{name}-M00Anim.fbx")
            _run([blender, "-b", *wincompat.blender_args(), "--python",
                  os.path.join(REPO, "win", "blender_make_wiggle.py"), "--", "--in", base_fbx,
                  "--out", out, "--mode", "keep"],
                 env, log, "1/9 (b) use the animation in the character FBX as the reference motion (Blender)")
            if os.path.isfile(out):
                log("   reference motion: the animation in the character FBX")
        if len(glob.glob(os.path.join(src_dir, "*.fbx"))) < 2:
            if not wiggle:
                raise RuntimeError("No animation found. Use a character FBX with an animation (any motion such as "
                                   "walk or idle), add a motion FBX, or turn on auto_wiggle.")
            # 실제 동작이 하나도 없을 때만: 모든 본을 살짝 움직이는 기준 모션 생성 (정확도는 실제 동작보다 낮음)
            out = os.path.join(src_dir, f"{name}-000Rest.fbx")
            _run([blender, "-b", *wincompat.blender_args(), "--python",
                  os.path.join(REPO, "win", "blender_make_wiggle.py"), "--", "--in", base_fbx, "--out", out],
                 env, log, "1/9 (b) build reference motion that moves every bone (Blender)")
            log("   [NOTE] no animation found - using an auto-generated reference motion. Results are more accurate "
                "with a character FBX that has a real animation (any walk or idle).")

        # 2) 무게 없는 끝 본 정리 (블렌더)
        stage_in = os.path.join(WORKSPACE, "_stage_in", name)
        shutil.rmtree(stage_in, ignore_errors=True)
        os.makedirs(stage_in)
        shutil.copytree(src_dir, os.path.join(stage_in, name))
        fixed_root = os.path.join(ZOO, "_fixed", name)
        _run(_py("preprocess/fix_fbx_batch.py", "--input_root", stage_in, "--output_root", fixed_root,
                 "--blender", blender, "--workers", env["MCA_WORKERS"]),
             env, log, "2/9 clean FBX - remove unweighted leaf bones (Blender)")
        if not glob.glob(os.path.join(fixed_root, name, "*.fbx")):
            raise RuntimeError("FBX cleanup produced no output. See zoo/_fixed/<name>/_fix_failed.tsv and the log.")

        # 3) 메시/스키닝/기본 포즈/모션 BVH 추출 (블렌더)
        env3 = dict(env, DATA_ROOT=fixed_root, ZOO_ROOT=ZOO)
        _run([blender, "-b", *wincompat.blender_args(), "--python",
              os.path.join(REPO, "preprocess", "extract_character_from_fbx.py")],
             env3, log, "3/9 extract mesh, skin weights and motions (Blender)")
        if not os.path.exists(os.path.join(ZOO, "characters_fix_facezplus", name, "base_mesh.obj")):
            raise RuntimeError("Mesh extraction failed. Make sure the character FBX contains both a mesh and an armature.")
        rest = os.path.join(ZOO, "characters_fix_facezplus", name, "rest.bvh")
        with open(rest, encoding="utf-8", errors="replace") as f:
            n_joints = sum(1 for ln in f if ln.strip().startswith(("ROOT ", "JOINT ")))
        log(f"   bones: {n_joints}")
        if n_joints > 150:
            raise RuntimeError(f"The rig has {n_joints} bones; MoCapAnything supports up to 150. "
                               "Remove physics bone chains (veil, hair, cloth) and use a body-only skeleton.")

        # 4) 정면을 +Z 로 정렬
        align_args = ["--input_root", "zoo/characters_fix_facezplus",
                      "--output_root", "zoo/characters_face_zplus",
                      "--motion_root", "zoo/motions", "--motion_output_root", "zoo/motions_face_zplus"]
        # 정면 방향은 항상 직접 지정 (자동 판별은 믹사모 캐릭터에서 90도 틀린 값을 낸 적이 있어 쓰지 않음)
        if front not in FRONT_CHOICES:
            raise RuntimeError(f"front must be one of {list(FRONT_CHOICES)}, got {front!r}")
        if True:
            import numpy as np
            ref_dir = os.path.join(WORKSPACE, "align_ref")
            os.makedirs(os.path.join(ref_dir, name), exist_ok=True)
            np.save(os.path.join(ref_dir, name, "front.npy"), np.array(FRONT_CHOICES[front], dtype=np.float64))
            align_args += ["--ref_dir", ref_dir]
        _run(_py("preprocess/align_character_face_zplus.py", *align_args), env, log, "4/9 align character to face +Z")
        if not os.path.exists(os.path.join(ZOO, "characters_face_zplus", name, "front.npy")):
            raise RuntimeError(
                "Alignment to +Z failed. See the log (zoo/characters_face_zplus/_align_failed.tsv).")

        # 5~8) 12방향 회전 / 포즈 npz / 종 정보(T5) / 스케일 캐시
        _run(_py("preprocess/rotate_bvh_parallel.py"), env, log, "5/9 rotate motions to 12 yaw views")
        _run(_py("preprocess/extract_bvh_pose.py"), env, log, "6/9 extract pose data")
        _run(_py("preprocess/build_species_info.py", "--dataset_root", "zoo", "--t5_model", model_store.t5_repo()), env, log,
             "7/9 skeleton topology + bone-name embeddings (T5)")
        _run(_py("preprocess/build_scale_cache.py", "--bvh_pose_root", "zoo/bvh_pose",
                 "--output", "zoo/cache/__mesh2pose1002_species_scale_cache.pkl"), env, log, "8/9 scale normalization")

        # 9) 참조 이미지: 정면(y0) 몇 프레임만 렌더 → 프레임 추출 → DINOv2 임베딩
        #    (추론은 참조 시퀀스의 첫 프레임만 쓰므로 12방향 전체를 렌더할 필요가 없음)
        _run(_py("preprocess/render_bvh_videos_fast.py", "--zoo-root", "zoo",
                 "--character-root", "zoo/characters_face_zplus",
                 "--scene", os.path.join(REPO, "preprocess", "blank.blend"),
                 "--blender", blender, "--views", "y0", "--max-frames", str(render_frames),
                 "--exact-motion-filter", _ref_motion(name), "--workers", "1", "--skip-existing"),
             env, log, "9/9 (a) render reference images (Blender EEVEE)")
        _make_thumb(name)
        _run(_py("preprocess/video_to_images.py"), env, log, "9/9 (b) extract frames")
        _run(_py("preprocess/preprocess_image_only.py", "--dataset_root", "zoo"), env, log,
             "9/9 (c) embed reference images (DINOv2 + background removal, GPU)")

        if name not in list_rigs():
            raise RuntimeError("Final check failed: some output files are missing. See the log.")
        log(f"Done: '{name}' prepared in {time.time() - t0:.0f} s. Select it in 'MCA Motion Capture'.")
        return name
    except Exception as e:
        log(f"[FAILED] {e}")
        raise
    finally:
        shutil.rmtree(os.path.join(WORKSPACE, "_stage_in", name), ignore_errors=True)
        log.close()


PREP_VERSION = 2   # 준비 방식이 바뀌면 올림 -> 예전에 준비한 캐릭터는 자동으로 다시 준비


def _state_path(name):
    return os.path.join(WORKSPACE, "prep_state", f"{name}.json")


def _signature(base_fbx, front):
    st = os.stat(base_fbx)
    return {"fbx": os.path.normcase(os.path.abspath(base_fbx)), "size": st.st_size,
            "mtime": int(st.st_mtime), "front": front, "version": PREP_VERSION}


def ref_image(name):
    p = os.path.join(ZOO, "ref_images", f"{name}.jpg")
    return p if os.path.exists(p) else ""


def prepare_if_changed(name, base_fbx, blender=None, front="+z", log_cb=None):
    """자동 변경 감지: 같은 이름이 같은 FBX(경로·크기·수정 시간)·같은 front 로 이미 준비돼 있으면 건너뜀.
    FBX 가 바뀌었거나, 같은 이름을 다른 FBX 가 쓰고 있었거나, 예전 버전에서 준비한 데이터면 다시 준비."""
    import json
    name = safe_name(name)
    if not base_fbx or not os.path.isfile(base_fbx):
        raise RuntimeError(f"Character FBX not found: {base_fbx}")
    sig = _signature(base_fbx, front)
    old = None
    try:
        with open(_state_path(name), encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        pass
    if old == sig and name in list_rigs():
        print(f"[prepare] '{name}' is unchanged (same FBX, front {front}) - preparation skipped", flush=True)
        return {"name": name, "skipped": True, "ref_image": ref_image(name)}
    if old and old.get("fbx") != sig["fbx"]:
        print(f"[prepare] NOTE: the name '{name}' was used by another FBX ({old.get('fbx')}) - it is replaced",
              flush=True)
    elif name in list_rigs() and not old:
        print(f"[prepare] '{name}' was prepared by an earlier version - preparing again", flush=True)
    try:
        os.remove(_state_path(name))
    except OSError:
        pass
    prepare_rig(name, base_fbx, (), blender, front, True, log_cb=log_cb)
    os.makedirs(os.path.dirname(_state_path(name)), exist_ok=True)
    with open(_state_path(name), "w", encoding="utf-8") as f:
        json.dump(sig, f)
    ref = ref_image(name)
    print(f"[prepare] front view check: {ref or '(no reference image)'} - the character must face the camera; "
          "if it shows the side or the back, choose another 'front' value", flush=True)
    return {"name": name, "skipped": False, "ref_image": ref}


def _make_thumb(name):
    """앱 갤러리용 썸네일: 정면 렌더 첫 프레임을 흰 배경에 얹어 zoo/ref_images/<name>.jpg 로 저장."""
    try:
        from PIL import Image
        cands = sorted(glob.glob(os.path.join(ZOO, "video_fast_work", "jobs", f"{name}#*", "y0", "images", "*.png")))
        if not cands:
            return
        im = Image.open(cands[0]).convert("RGBA")
        bbox = im.getchannel("A").getbbox()
        if bbox:
            pad = 24
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
            im = im.crop(bbox)
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        out = os.path.join(ZOO, "ref_images", f"{name}.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bg.convert("RGB").save(out, quality=92)
    except Exception as e:  # 썸네일은 없어도 동작에 지장 없음
        print(f"[thumb] skipped: {e}")


def _ref_motion(name):
    """참조로 쓸 시퀀스(정렬상 첫 번째 모션) 폴더 이름: 예) Beatrice#Beatrice-000Rest"""
    cands = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ZOO, "bvh", f"{name}#*")))
    if not cands:
        raise RuntimeError("No rotated motions (BVH). See the earlier log lines.")
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="MoCapAnything - prepare your own character (Windows)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="character FBX (mesh + armature + skin)")
    ap.add_argument("--motion", action="append", default=[], help="motion FBX (repeatable)")
    ap.add_argument("--blender")
    ap.add_argument("--front", default="+z", choices=list(FRONT_CHOICES))
    ap.add_argument("--no-wiggle", action="store_true", help="do not auto-generate the reference motion")
    a = ap.parse_args()
    prepare_rig(a.name, a.base, a.motion, a.blender, a.front, not a.no_wiggle)


if __name__ == "__main__":
    main()
