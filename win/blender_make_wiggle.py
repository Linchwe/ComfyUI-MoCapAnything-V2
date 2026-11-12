"""(블렌더 내부에서 실행) 기본 포즈만 있는 캐릭터 FBX 로 '모든 본이 조금씩 움직이는' 모션 FBX 를 만듭니다.

왜 필요한가
 - MoCapAnything 전처리는 모션 데이터에서 한 번도 회전하지 않은 본을 '고정 본'으로 기록하고,
   추론할 때 그 본은 회전을 만들지 않습니다.
 - 기본 포즈만 넣으면 모든 본이 고정으로 잡혀 결과 모션이 굳어 버립니다.
 - 그래서 첫 프레임은 기본 포즈 그대로 두고, 이후 프레임에서 모든 본을 작은 각도로 흔든 모션을 만듭니다.
   (첫 프레임이 기본 포즈라 추론 시 참조 포즈도 기본 포즈가 됩니다)

모드
 - wiggle : 위 설명의 기준 모션
 - rest   : 기본 포즈 그대로인 2프레임 액션을 붙인 캐릭터 FBX (원본 FBX 정리 단계가 액션 없는 FBX 를 처리하지 못함)
 - keep   : 사용자 모션 FBX 의 애니메이션은 그대로 두고 단위/본 이름 규칙만 맞춤

모든 모드에서 공통으로
 - 단위 규칙을 '아마추어 스케일 0.01 + 데이터 cm' 로 맞춤 (normalize_units)
 - 공백이 들어간 본 이름을 _ 로 바꿈 (BVH 형식은 이름에 공백을 허용하지 않음)
캐릭터·기준 모션·사용자 모션이 모두 이 스크립트를 거치므로 서로 같은 규칙이 됩니다.

실행: blender -b --python blender_make_wiggle.py -- --in a.fbx --out b.fbx [--mode wiggle|rest|keep] [--frames 48] [--deg 12]
"""
import argparse
import math
import re
import sys

import bpy


MIN_ANIM_FRAMES = 10   # 이보다 짧으면 진짜 애니메이션으로 보지 않음 (T포즈 + 원래 자세 같은 짧은 FBX)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--deg", type=float, default=12.0)
    ap.add_argument("--mode", choices=["wiggle", "rest", "keep"], default="wiggle")
    return ap.parse_args(argv)


def normalize_units(arm):
    """단위 규칙 맞추기: '아마추어 오브젝트 스케일 0.01 + 데이터는 cm' (Mixamo·Truebones FBX 를
    블렌더로 불러왔을 때의 상태). 원본 전처리가 이 규칙(BVH 오프셋이 메시보다 100배 큼)을 전제로 합니다.
    블렌더에서 미터 단위로 내보낸 FBX 는 불러오면 스케일 1.0 이라, 여기서 같은 규칙으로 바꿉니다.
    보이는 크기와 회전은 그대로 유지됩니다."""
    s = arm.matrix_world.to_scale().x
    if abs(s - 0.01) <= 0.0005:
        print(f"[units] already cm convention (scale={s:.4f})")
        return 1.0
    print(f"[units] armature scale {s:.4f} -> converting to the 0.01 (cm) convention")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for m in meshes:   # 스킨 메시는 아마추어 자식으로 (변환을 같이 따라가게)
        if m.parent is not arm:
            mw = m.matrix_world.copy()
            m.parent = arm
            m.matrix_world = mw
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None

    def apply_scale():
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    apply_scale()                       # 데이터 = 실제 크기(m), 스케일 1
    arm.scale = (100.0, 100.0, 100.0)
    apply_scale()                       # 데이터 = cm 값, 스케일 1
    arm.scale = (0.01, 0.01, 0.01)      # 보이는 크기는 원래대로
    bpy.context.view_layer.update()
    return s * 100.0                    # 뼈대 데이터가 커진 배율 (모션의 위치 키도 같은 배율로)


def to_y_up(arm):
    """뼈대 데이터의 위쪽을 +Y 로 맞추기 (믹사모 FBX 를 불러왔을 때와 같은 규칙: 아마추어 오브젝트를 X 로 90도
    세우고, 뼈대 데이터는 Y 가 위). 블렌더 등에서 뼈대 데이터가 Z 위로 들어오는 FBX 가 있는데, 원본 전처리는
    BVH 의 Y 가 위라고 가정합니다. 보이는 모습과 애니메이션은 그대로 유지됩니다."""
    from mathutils import Matrix, Vector

    bones = arm.data.bones
    head = next((b for b in bones if "head" in b.name.lower()
                 and not any(x in b.name.lower() for x in ("top", "end", "nub"))), None)
    roots = [b for b in bones if b.parent is None]
    if head is None or not roots:
        return False
    v = head.head_local - roots[0].head_local
    k = max(range(3), key=lambda i: abs(v[i]))
    up = Vector([0.0, 0.0, 0.0])
    up[k] = 1.0 if v[k] > 0 else -1.0
    if k == 1 and v[k] > 0:
        return False
    print(f"[axis] skeleton data is {'+-'[v[k] < 0]}{'XYZ'[k]}-up -> converting to Y-up (look unchanged)")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for m in meshes:   # 스킨 메시는 아마추어 자식으로 (변환을 같이 따라가게)
        if m.parent is not arm:
            mw = m.matrix_world.copy()
            m.parent = arm
            m.matrix_world = mw
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    # 아마추어 오브젝트 자체의 회전/스케일 키(블렌더에서 구운 FBX 에 생김)는 아래에서 되돌리는 회전을 덮어쓰므로 지움
    act = arm.animation_data.action if arm.animation_data else None
    if act is not None:
        bags = [act]
        if not hasattr(act, "fcurves"):
            bags = [cb for ly in act.layers for st in ly.strips for cb in getattr(st, "channelbags", [])]
        for bag in bags:
            for fc in [c for c in bag.fcurves if c.data_path in ("rotation_euler", "rotation_quaternion",
                                                                  "rotation_axis_angle", "scale")]:
                bag.fcurves.remove(fc)
    rot = up.rotation_difference(Vector((0.0, 1.0, 0.0))).to_matrix().to_4x4()
    arm.matrix_world = rot @ arm.matrix_world           # 데이터를 돌릴 만큼 오브젝트를 먼저 돌리고
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)   # 그 회전을 데이터에 굽고
    arm.matrix_world = rot.inverted() @ arm.matrix_world  # 오브젝트는 반대로 돌려 보이는 모습은 원래대로
    bpy.context.view_layer.update()
    return True


def _fcurves(action):
    """블렌더 4.4+ 슬롯 액션/이전 액션 모두에서 F-Curve 목록 얻기."""
    try:
        return list(action.fcurves)
    except AttributeError:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out += list(bag.fcurves)
        return out


def scale_location_keys(arm, factor):
    """뼈대 데이터 배율이 바뀐 만큼 본 위치(location) 키 값을 같이 키움 (루트 이동 등)."""
    if abs(factor - 1.0) < 1e-6 or not (arm.animation_data and arm.animation_data.action):
        return
    n = 0
    for fc in _fcurves(arm.animation_data.action):
        if fc.data_path.startswith("pose.bones[") and fc.data_path.endswith(".location"):
            for kp in fc.keyframe_points:
                kp.co.y *= factor
                kp.handle_left.y *= factor
                kp.handle_right.y *= factor
            n += 1
    print(f"[units] scaled {n} location curves x{factor:g}")


def sanitize_bone_names(arm):
    """공백이 든 본 이름 → _ (버텍스 그룹·애니메이션 경로는 블렌더가 같이 바꿔 줌)."""
    n = 0
    for b in arm.data.bones:
        if re.search(r"\s", b.name):
            b.name = re.sub(r"\s+", "_", b.name.strip())
            n += 1
    if n:
        print(f"[names] replaced spaces with _ in {n} bone names")


def main():
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=args.inp)

    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("[wiggle] the FBX has no armature.")

    factor = normalize_units(arm)
    to_y_up(arm)
    sanitize_bone_names(arm)

    if args.mode == "keep":
        scale_location_keys(arm, factor)
        act = arm.animation_data.action if arm.animation_data else None
        if act is None:
            print("[keep] no animation in this FBX - nothing written")
            return
        f0, f1 = (int(round(x)) for x in act.frame_range)
        # MoCapAnything uses the FIRST frame of the reference motion as its reference pose. Exporters often put
        # the T-pose / bind pose on the first frame(s) (e.g. a T-pose first frame), so those are skipped.
        scene = bpy.context.scene
        moved, start = 0.0, None
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            dev = max((pb.matrix_basis.to_quaternion().angle for pb in arm.pose.bones), default=0.0)
            moved = max(moved, dev)
            if start is None and math.degrees(dev) > 5.0:
                start = f
        if math.degrees(moved) < 2.0:
            print("[keep] the animation does not move (rest pose only) - nothing written")
            return
        start = f0 if start is None else start
        if start > f0:
            print(f"[keep] skipped {start - f0} rest/T-pose frame(s) at the start")
        if f1 - start + 1 < MIN_ANIM_FRAMES:
            # 바인드 포즈 + 원래 자세 몇 프레임뿐인 FBX 는 진짜 애니메이션이 아님. 이걸 참조로 쓰면
            # 거의 모든 뼈가 '고정 뼈'로 잡혀 결과가 굳으므로, 쓰지 않고 auto_wiggle(모든 뼈를 움직이는 기준 모션)로 넘김.
            print(f"[keep] only {f1 - start + 1} frame(s) after the T-pose - not a real animation, nothing written")
            return
        scene.frame_start, scene.frame_end = start, f1
        export(args.out)
        print(f"[keep] {f1 - start + 1} frames -> {args.out}")
        return

    # 기존 액션은 지우고 새로 만듦
    if arm.animation_data:
        arm.animation_data_clear()
    arm.animation_data_create()
    action = bpy.data.actions.new(name="MCA_Wiggle")
    arm.animation_data.action = action

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")

    n = 2 if args.mode == "rest" else max(8, args.frames)
    amp = math.radians(args.deg)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 30, 1.0   # 원본 전처리는 30fps 기준 (60fps FBX 의 2프레임 -> 1프레임 방지)
    scene.frame_start = 1
    scene.frame_end = n

    bones = list(arm.pose.bones)
    for pb in bones:
        pb.rotation_mode = "XYZ"
        pb.location = (0.0, 0.0, 0.0)

    for f in range(1, n + 1):
        t = (f - 1) / (n - 1)
        for i, pb in enumerate(bones):
            if f == 1 or pb.parent is None or args.mode == "rest":
                rot = (0.0, 0.0, 0.0)   # 1프레임 = 기본 포즈, 루트는 고정
            else:
                ph = i * 0.73
                rot = (
                    amp * math.sin(2 * math.pi * (t * 2) + ph),
                    amp * 0.6 * math.sin(2 * math.pi * (t * 3) + ph * 1.7),
                    amp * math.sin(2 * math.pi * (t * 1) + ph * 2.3),
                )
                rot = tuple(r * math.sin(math.pi * t) for r in rot)  # 시작/끝은 0 으로 부드럽게
            pb.rotation_euler = rot
            pb.keyframe_insert("rotation_euler", frame=f)

    bpy.ops.object.mode_set(mode="OBJECT")
    export(args.out)
    print(f"[{args.mode}] {len(bones)} bones, {n} frames -> {args.out}")


def export(path):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
    )


main()
sys.stdout.flush()
