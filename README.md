# ComfyUI-MoCapAnything-V2

Custom nodes that run [MoCapAnything V2](https://github.com/phongdaot/MocapAnything) (MIT) in ComfyUI portable on Windows.
Captures the motion in a video **on your own character's skeleton** and saves one FBX, GLB or BVH file.

- Install: put this folder in `ComfyUI/custom_nodes/`, install the packages in `requirements.txt` (ComfyUI Manager does this automatically) and restart ComfyUI.
- Models: downloaded automatically on first run to `ComfyUI/models/mocapanything/` (about 4 GB).
- Requires: NVIDIA GPU, Blender 4.2+ (character preparation only).
- Nodes: `MCA Prepare Character` (outputs the target name and a front-view image), `MCA Motion Capture`.
- Example workflow: Templates > EXTENSIONS > ComfyUI-MoCapAnything-V2 (`example_workflows/MocapAnything_video_to_motion.json`).
- Background removal: the original RMBG-1.4 (non-commercial) is replaced with BiRefNet (MIT).

Manual: `MANUAL_EN.txt` (English), `MANUAL_KO.txt` (Korean). Original README: `README_ORIGINAL.md`.

---

[MoCapAnything V2](https://github.com/phongdaot/MocapAnything) (MIT) 를 윈도우 ComfyUI 포터블에서 쓰도록 옮긴 커스텀 노드입니다.
영상 속 동작을 **내가 만든 캐릭터의 뼈대 그대로** FBX, GLB 또는 BVH 파일 하나로 뽑습니다. 자세한 내용은 `MANUAL_KO.txt` 를 보세요.
