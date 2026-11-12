"""ComfyUI-MoCapAnything-V2 — MoCapAnything V2 for ComfyUI (Windows / ComfyUI portable)

Nodes (category: MocapAnything)
 - MCA Prepare Character : rigged character FBX -> MoCapAnything data. Prepares again only when the FBX
                           (or front) changed; outputs the character name and its front-view reference image
 - MCA Motion Capture    : video -> FBX / GLB / BVH on that character + 3D previews

Design notes
 - All heavy work runs in a separate Python process (win/mocap_cli.py). This repo's `utils` / `data` /
   `models` packages clash with ComfyUI's own module names, so importing them into the ComfyUI process
   could break ComfyUI.
 - Therefore this file only uses the standard library and ComfyUI modules (never import wincompat etc. here).
 - Before running, ComfyUI's models can be unloaded from VRAM (free_comfy_vram); when the subprocess exits,
   all VRAM used by MoCapAnything is released.
"""
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "win", "mocap_cli.py")
try:
    import folder_paths

    # register ComfyUI/models/mocapanything like other nodes (extra_model_paths.yaml "mocapanything" also works)
    folder_paths.add_model_folder_path("mocapanything", os.path.join(folder_paths.models_dir, "mocapanything"))
    MODELS_DIR = folder_paths.get_folder_paths("mocapanything")[0]
    OUTPUT_DIR = folder_paths.get_output_directory()
    INPUT_DIR = folder_paths.get_input_directory()
    TEMP_DIR = folder_paths.get_temp_directory()
except Exception:  # imported outside ComfyUI (tests)
    folder_paths = None
    MODELS_DIR = os.path.join(HERE, "_models")
    OUTPUT_DIR = os.path.join(HERE, "outputs")
    INPUT_DIR = os.path.join(HERE, "inputs")
    TEMP_DIR = os.path.join(HERE, "temp")

# models: ComfyUI/models/mocapanything/{checkpoints, hf}   prepared characters: .../mocapanything/characters
MODELS_DIR = os.environ.get("MCA_MODELS_DIR", MODELS_DIR)
WORKSPACE = os.environ.get("MCA_WORKSPACE", os.path.join(MODELS_DIR, "characters"))
ZOO = os.path.join(WORKSPACE, "zoo")

try:  # recent ComfyUI: native 3D / VIDEO outputs connect straight to Preview 3D / Save Video
    from comfy_api.latest import InputImpl, Types

    HAS_API = True
except Exception:
    HAS_API = False

FRONT = ["+z", "-z", "+x", "-x"]
NO_FBX = "(no FBX in the input folder - use fbx_path)"


# ----------------------------------------------------------------------------- helpers
SAVE_FORMATS = ["FBX", "GLB", "BVH"]
# clean-up options (same names as win/motion_fix.py)
STABILIZE = ["fix flips", "lock facing", "off"]
MIRROR_SIDES = ["off", "auto", "left -> right", "right -> left"]
MIRROR_TIMING = ["match original", "half cycle", "same frame"]


def _output_file(prefix, ext):
    """Same naming as ComfyUI's Save nodes: output/<subfolder>/<name>_00001_.<ext> (never overwrites)."""
    prefix = prefix.replace("\\", "/").strip().strip("/") or "MocapAnything/Mocap"
    if folder_paths is not None:
        folder, name, counter, _sub, _ = folder_paths.get_save_image_path(prefix, OUTPUT_DIR)
    else:
        folder = os.path.join(OUTPUT_DIR, os.path.dirname(prefix))
        name, counter = os.path.basename(prefix), 1
    os.makedirs(folder, exist_ok=True)
    while True:   # the counter scan only looks at image-like names, so double-check for our extension
        path = os.path.join(folder, f"{name}_{counter:05}_.{ext}")
        if not os.path.exists(path):
            return path
        counter += 1


def _safe_name(name):
    """Same rule as win/rig_prep.safe_name (copied because this file must not import repo modules)."""
    s = re.sub(r"[^A-Za-z0-9\-]+", "", (name or "").replace(" ", "-").replace("_", "-"))
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "MyRig"


def _prepared_targets():
    out = []
    for d in sorted(glob.glob(os.path.join(ZOO, "characters_face_zplus", "*"))):
        name = os.path.basename(d)
        if (os.path.exists(os.path.join(d, "base_mesh.obj"))
                and glob.glob(os.path.join(ZOO, "npz_train_image_only", f"{name}#*", "y0.npz"))):
            out.append(name)
    return out


def _input_fbx_files():
    """FBX files in the ComfyUI input folder, including subfolders (e.g. input/3d from Load 3D)."""
    files = []
    for p in glob.glob(os.path.join(INPUT_DIR, "**", "*.fbx"), recursive=True) + \
            glob.glob(os.path.join(INPUT_DIR, "**", "*.FBX"), recursive=True):
        files.append(os.path.relpath(p, INPUT_DIR).replace("\\", "/"))
    return sorted(set(files))


def _resolve_input_path(p):
    p = (p or "").strip().strip('"')
    if not p:
        return ""
    if not os.path.isabs(p):
        cand = os.path.join(INPUT_DIR, p)
        if os.path.exists(cand):
            return cand
    return p


def _free_comfy_vram():
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(True)
        print("[MocapAnything] Unloaded ComfyUI models from VRAM.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[MocapAnything] Skipped freeing VRAM: {e}", flush=True)


def _kill_tree(p):
    try:
        if os.name == "nt":  # also kill grandchildren such as Blender
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        p.wait(timeout=10)
    except Exception:
        pass


def _run_cli(args, progress_map=None):
    """Run win/mocap_cli.py, stream its log to the console, drive the progress bar, honour Cancel."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MCA_MODELS_DIR"] = MODELS_DIR
    env["MCA_WORKSPACE"] = WORKSPACE
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    cmd = [sys.executable, "-s", CLI, *args]
    print("[MocapAnything] run:", " ".join(f'"{a}"' if " " in a else a for a in cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    q = queue.Queue()

    def reader():
        for line in p.stdout:
            q.put(line.rstrip("\r\n"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    pbar, mm = None, None
    try:
        import comfy.model_management as mm  # noqa: F811
        import comfy.utils

        pbar = comfy.utils.ProgressBar(100)
    except Exception:
        pass

    result, tail, done = None, [], False
    while not done:
        if mm is not None and mm.processing_interrupted():
            _kill_tree(p)
            mm.throw_exception_if_processing_interrupted()
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            done = True
            break
        if line.startswith("MCA_RESULT "):
            result = json.loads(line[len("MCA_RESULT "):])
            continue
        if not line.strip() or "it/s]" in line or "s/it]" in line:
            continue
        tail = (tail + [line])[-40:]
        print("[MocapAnything]", line, flush=True)
        if pbar is not None and progress_map:
            for key, val in progress_map:
                if key in line:
                    pbar.update_absolute(val, 100)
                    break
    rc = p.wait()
    if rc != 0 or result is None:
        raise RuntimeError("MocapAnything failed - see the ComfyUI console. Last log lines:\n" + "\n".join(tail[-15:]))
    if pbar is not None:
        pbar.update_absolute(100, 100)
    return result


PREP_PROGRESS = ([("[models] downloading", 2), ("[prepare]", 1)] + [(f"── {i}/9", int(i * 100 / 10)) for i in range(1, 10)]
                 + [("Done:", 100)])
MOCAP_PROGRESS = [("[models] downloading", 3), ("loading models", 10), ("models ready", 25), ("frames ->", 35),
                  ("stage: dino", 45), ("stage: v2p", 60), ("stage: p2r", 70), ("stage: plot", 80),
                  ("[motion]", 85), ("stage: export", 90), ("[engine] done", 100)]


# ----------------------------------------------------------------------------- nodes
class MCAPrepareRig:
    CATEGORY = "MocapAnything"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("target", "front_view")
    FUNCTION = "run"
    DESCRIPTION = ("Prepares a rigged character FBX (mesh + armature + skin weights) for MoCapAnything. It runs again "
                   "only when the FBX file (or front) changed - otherwise it returns at once. If the FBX holds an "
                   "animation (any walk or idle, e.g. Mixamo 'With Skin'), its first moving frame is the reference "
                   "pose; without one a reference motion is generated. front_view shows the reference render: the "
                   "character must face the camera - if it shows the side or the back, change 'front'.")

    @classmethod
    def INPUT_TYPES(cls):
        files = _input_fbx_files() or [NO_FBX]
        return {
            "required": {
                "fbx_file": (files, {"tooltip": "Character FBX from the ComfyUI input folder (subfolders included, e.g. input/3d)"}),
                "name": ("STRING", {"default": "", "tooltip": "Character name (letters/digits). Empty = FBX file name. "
                                    "Mixamo files are often named after the animation (Walking.fbx) - give each character its own name"}),
                "front": (FRONT, {"default": "+z", "tooltip":
                                  "Direction the character faces in the FBX. +z = Mixamo, and Blender rigs whose face is "
                                  "visible in the Front view (default FBX export). Check front_view"}),
            },
            "optional": {
                "fbx_path": ("STRING", {"default": "", "tooltip": "Full path to the character FBX (overrides fbx_file)"}),
                "blender_path": ("STRING", {"default": "", "tooltip": "Path to blender.exe. Empty = auto-detect"}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, fbx_file):
        return True

    def run(self, fbx_file, name, front, fbx_path="", blender_path=""):
        base = _resolve_input_path(fbx_path) or (_resolve_input_path(fbx_file) if fbx_file != NO_FBX else "")
        if not base or not os.path.isfile(base):
            raise RuntimeError(f"Character FBX not found: {base or '(empty)'}")
        target = _safe_name(name or os.path.splitext(os.path.basename(base))[0])
        args = ["prepare", "--name", target, "--base", base, "--front", front]
        if (blender_path or "").strip():
            args += ["--blender", blender_path.strip().strip('"')]
        res = _run_cli(args, PREP_PROGRESS)
        return (res["name"], _load_image(res.get("ref_image", "")))


def _load_image(path):
    """Image file -> ComfyUI IMAGE [1,H,W,3] (a small gray image when missing)."""
    import numpy as np
    import torch

    try:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    except Exception:
        arr = np.full((64, 64, 3), 0.5, np.float32)
    return torch.from_numpy(arr)[None]


class MCAMocap:
    CATEGORY = "MocapAnything"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "FILE_3D_GLB" if HAS_API else "STRING", "FILE_3D_GLB" if HAS_API else "STRING")
    RETURN_NAMES = ("file_path", "mesh_3d", "skeleton_3d")
    DESCRIPTION = ("Captures the motion in a video onto the character from MCA Prepare Character (connect its 'target' "
                   "output) and saves ONE file in the output folder: FBX or GLB (rigged mesh + skeleton + animation) "
                   "or BVH (skeleton + animation). Bone names and rest pose are the character's own. mesh_3d / "
                   "skeleton_3d are preview-only files for 'Preview 3D (Advanced)'. Input is resampled to 30 fps and "
                   "up to 10 seconds are processed (model limit: 301 frames).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("STRING", {"forceInput": True, "tooltip": "Connect the 'target' output of MCA Prepare Character"}),
                "filename_prefix": ("STRING", {"default": "MocapAnything/Mocap", "tooltip":
                                               "Path and file name in the output folder, like Save nodes. "
                                               "Meshes/RigMesh -> output/Meshes/RigMesh_00001_.fbx"}),
                "save_format": (SAVE_FORMATS, {"default": "FBX", "tooltip":
                                               "FBX / GLB: rigged mesh + skeleton + animation in one file. "
                                               "BVH: skeleton + animation only (the format cannot hold a mesh)"}),
                "use_sam": ("BOOLEAN", {"default": True, "tooltip":
                                        "Isolate the main person with SAM2 (recommended for busy backgrounds or "
                                        "several people). Off = BiRefNet background removal only"}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                                            "tooltip": "Start time in the video (seconds)"}),
                "max_seconds": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 10.0, "step": 0.1,
                                          "tooltip": "Length to process (seconds). Max 10 s (model limit)"}),
                "stabilize": (STABILIZE, {"default": "fix flips", "tooltip":
                              "fix flips: repair single frames where a joint snaps away and back. "
                              "lock facing: also remove the body's turning, so the character always faces its "
                              "own front (front-facing dances, walks). off: raw model output"}),
                "foot_lock": ("BOOLEAN", {"default": False, "tooltip":
                              "Put the lowest foot on the ground every frame and keep the standing foot from "
                              "sliding by moving the hips (the character travels with its steps). Good for walks / "
                              "runs; keep it off for jumps, sitting or kneeling"}),
                "in_place": ("BOOLEAN", {"default": False, "tooltip":
                             "Remove travel on the ground, like Mixamo's In Place. With foot_lock the up/down motion "
                             "is kept; without it the hips stay at one spot"}),
                "mirror_limbs": (MIRROR_SIDES, {"default": "off", "tooltip":
                                 "Copy the arm/leg motion of one side to the other, mirrored (side views where "
                                 "the far side is hidden and limps). auto = the steadier side. Needs Left/Right "
                                 "(or L/R) in the bone names"}),
                "mirror_timing": (MIRROR_TIMING, {"default": "match original", "tooltip":
                                  "match original: line the copy up with the beats of the side it replaces. "
                                  "half cycle: shift by half a step cycle. same frame: no shift (jumps, "
                                  "symmetric moves)"}),
                "keep_work_files": ("BOOLEAN", {"default": False, "tooltip":
                                                "Keep intermediate files (frames, raw BVH) in ComfyUI/temp/mocapanything "
                                                "(the temp folder is cleared when ComfyUI starts)"}),
                "free_comfy_vram": ("BOOLEAN", {"default": True, "tooltip": "Unload ComfyUI's models from VRAM before running"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Output of a Load Video node"}),
                "images": ("IMAGE", {"tooltip": "Image sequence (frame batch) instead of a VIDEO"}),
                "images_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "tooltip": "Frame rate of the images input"}),
                "video_path": ("STRING", {"default": "", "tooltip": "Video file path instead of video/images (relative to the input folder allowed)"}),
            },
        }

    def run(self, target, filename_prefix, save_format, use_sam, start_seconds, max_seconds, stabilize, foot_lock,
            in_place, mirror_limbs, mirror_timing, keep_work_files, free_comfy_vram, video=None, images=None,
            images_fps=30.0, video_path=""):
        tgt = (target or "").strip()
        if tgt not in _prepared_targets():
            raise RuntimeError(f"'{tgt}' is not a prepared character. Connect the 'target' output of "
                               "MCA Prepare Character.")
        fmt = (save_format or "FBX").lower()
        out_file = _output_file(filename_prefix or "MocapAnything/Mocap", fmt)
        work_root = os.path.join(TEMP_DIR, "mocapanything")
        os.makedirs(work_root, exist_ok=True)
        src, tmp_src = self._input_video(video, images, images_fps, video_path, work_root)
        try:
            if free_comfy_vram:
                _free_comfy_vram()
            args = ["mocap", "--video", src, "--target", tgt, "--out-file", out_file, "--format", fmt,
                    "--work-root", work_root, "--start", str(start_seconds), "--seconds", str(max_seconds)]
            if use_sam:
                args.append("--sam")
            if keep_work_files:
                args.append("--keep-work")
            args += ["--stabilize", stabilize, "--mirror", mirror_limbs, "--mirror-timing", mirror_timing]
            if foot_lock:
                args.append("--foot-lock")
            if in_place:
                args.append("--in-place")
            res = _run_cli(args, MOCAP_PROGRESS)
        finally:
            if tmp_src and os.path.exists(tmp_src):
                os.remove(tmp_src)

        mesh, skel = res.get("mesh_glb", ""), res.get("skeleton_glb", "")
        if HAS_API:
            mesh_o = Types.File3D(mesh) if mesh else None
            skel_o = Types.File3D(skel) if skel else None
        else:
            mesh_o, skel_o = mesh, skel
        summary = f"Saved: {res.get('file', '')}\nFrames: {res.get('frames', '?')}"
        return {"ui": {"text": [summary]}, "result": (res.get("file", ""), mesh_o, skel_o)}

    @staticmethod
    def _input_video(video, images, images_fps, video_path, out_dir):
        """Return (video file path, temp file to delete afterwards or None)."""
        if video is not None:
            try:
                srcp = video.get_stream_source()
                if isinstance(srcp, str) and os.path.isfile(srcp):
                    return srcp, None
            except Exception:
                pass
            tmp = os.path.join(out_dir, f"_input_{int(time.time() * 1000)}.mp4")
            video.save_to(tmp)
            return tmp, tmp
        if images is not None:
            import imageio_ffmpeg
            import numpy as np

            arr = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)   # [N,H,W,C]
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            h, w = arr.shape[1] - arr.shape[1] % 2, arr.shape[2] - arr.shape[2] % 2
            tmp = os.path.join(out_dir, f"_input_{int(time.time() * 1000)}.mp4")
            wr = imageio_ffmpeg.write_frames(tmp, (w, h), fps=float(images_fps), macro_block_size=1,
                                             pix_fmt_out="yuv420p", quality=9)
            wr.send(None)
            for fr in arr:
                wr.send(np.ascontiguousarray(fr[:h, :w]))
            wr.close()
            return tmp, tmp
        p = _resolve_input_path(video_path)
        if p and os.path.isfile(p):
            return p, None
        raise RuntimeError("No input video. Connect video (Load Video), images, or set video_path.")


NODE_CLASS_MAPPINGS = {
    "MCAPrepareRig": MCAPrepareRig,
    "MCAMocap": MCAMocap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MCAPrepareRig": "MCA Prepare Character",
    "MCAMocap": "MCA Motion Capture",
}
