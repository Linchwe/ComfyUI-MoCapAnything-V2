"""명령줄 실행기 — ComfyUI 커스텀 노드가 이 파일을 별도 프로세스로 실행합니다.

(같은 프로세스에서 import 하면 이 저장소의 utils/data/models 패키지가
 ComfyUI 의 같은 이름 패키지와 충돌하기 때문에 프로세스를 분리합니다.)

  python win/mocap_cli.py list
  python win/mocap_cli.py mocap  --video in.mp4 --target Beatrice --out-file D:/out/a.fbx --format fbx [--sam] [--start 0] [--seconds 10] [--keep-work]
                                 [--stabilize "fix flips"] [--mirror off] [--mirror-timing "match original"] [--foot-lock] [--in-place]
  python win/mocap_cli.py prepare --name Beatrice --base char.fbx [--blender ...] [--front +z]
      (prepares only when the FBX or front changed; otherwise returns at once)

마지막 줄에 결과를 'MCA_RESULT {json}' 형식으로 출력합니다.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wincompat  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    m = sub.add_parser("mocap")
    m.add_argument("--video", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--out-file", required=True)
    m.add_argument("--format", choices=["fbx", "glb", "bvh"], default="fbx")
    m.add_argument("--work-root")
    m.add_argument("--sam", action="store_true")
    m.add_argument("--start", type=float, default=0.0)
    m.add_argument("--seconds", type=float, default=10.0)
    m.add_argument("--keep-work", action="store_true")
    m.add_argument("--stabilize", default="fix flips")
    m.add_argument("--mirror", default="off")
    m.add_argument("--mirror-timing", default="match original")
    m.add_argument("--foot-lock", action="store_true")
    m.add_argument("--in-place", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--blender")
    p.add_argument("--front", default="+z", choices=["+z", "-z", "+x", "-x"])
    a = ap.parse_args()

    if a.cmd == "list":
        from win.engine import list_targets
        res = {"targets": list_targets()}
    elif a.cmd == "mocap":
        from win.engine import MocapEngine
        res = MocapEngine().run(a.video, a.target, a.out_file, fmt=a.format, work_root=a.work_root, use_sam=a.sam,
                                start=a.start, seconds=a.seconds, keep_work=a.keep_work, stabilize=a.stabilize,
                                mirror_side=a.mirror, mirror_timing=a.mirror_timing, foot_lock=a.foot_lock,
                                in_place=a.in_place)
    else:
        from win import rig_prep
        res = rig_prep.prepare_if_changed(a.name, a.base, a.blender, a.front)
    print("MCA_RESULT " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
