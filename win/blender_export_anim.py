"""(Runs inside Blender) Put a MoCapAnything BVH motion onto the prepared character and save ONE file.

The result keeps the character's own mesh, armature (bone names, rest pose, hierarchy) and skinning,
with the captured motion as a single baked action.

How the motion is transferred
 - The BVH skeleton is the character's skeleton (same bone names), but it may live in another space
   (after the "face +Z" alignment, other axes/units). The rest joint positions of both skeletons are
   matched with a similarity transform (rotation + uniform scale + offset).
 - Every frame, each joint's world rotation relative to its BVH rest is mapped into armature space and
   applied on top of the bone's rest orientation; the root also gets the mapped translation.
 - Bones missing from the BVH (e.g. unweighted end bones) simply follow their parent.

Run: blender -b --python blender_export_anim.py -- --char char.fbx --bvh motion.bvh --out out.fbx
             [--format fbx|glb] [--fps 30]
"""
import argparse
import math
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["fbx", "glb"], default="fbx")
    ap.add_argument("--fps", type=int, default=30)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- BVH
def read_bvh(path):
    """Returns joints [(name, parent_index, offset(3), channels[list])], frame data (F, C), frame_time."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    head, motion = re.split(r"\bMOTION\b", text, maxsplit=1)
    tokens = head.split()
    joints, stack, i, last = [], [], 0, None
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append([tokens[i + 1], stack[-1] if stack else -1, None, []])
            last = len(joints) - 1
            i += 2
        elif t == "End":                       # End Site { OFFSET x y z }
            depth, i = 0, i + 2
            while True:
                if tokens[i] == "{":
                    depth += 1
                elif tokens[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            i += 1
        elif t == "{":
            stack.append(last)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[last][2] = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            joints[last][3] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    lines = [ln for ln in motion.strip().splitlines() if ln.strip()]
    n_frames = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    data = np.array([[float(x) for x in ln.split()] for ln in lines[2:2 + n_frames]], dtype=np.float64)
    return joints, data, frame_time


def _axis_rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bvh_fk(joints, data):
    """World rotations (F, J, 3, 3) and positions (F, J, 3); rest positions (J, 3)."""
    F, J = data.shape[0], len(joints)
    rots = np.zeros((F, J, 3, 3))
    pos = np.zeros((F, J, 3))
    rest = np.zeros((J, 3))
    col = 0
    cols = []
    for name, parent, off, ch in joints:
        cols.append(col)
        col += len(ch)
    for j, (name, parent, off, ch) in enumerate(joints):
        rest[j] = off if parent < 0 else rest[parent] + off
    for f in range(F):
        for j, (name, parent, off, ch) in enumerate(joints):
            local_r = np.eye(3)
            local_t = off.copy()
            vals = data[f, cols[j]:cols[j] + len(ch)]
            for c, v in zip(ch, vals):
                if c.endswith("rotation"):
                    local_r = local_r @ _axis_rot(c[0].upper(), v)
                elif c.endswith("position"):
                    local_t["XYZ".index(c[0].upper())] = v
            if parent < 0:
                rots[f, j] = local_r
                pos[f, j] = local_t
            else:
                rots[f, j] = rots[f, parent] @ local_r
                pos[f, j] = pos[f, parent] + rots[f, parent] @ local_t
    return rots, pos, rest


def similarity(src, dst):
    """dst ~= s * R @ src + t (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return R, s, t


# ----------------------------------------------------------------------------- scene
def _fcurves(action):
    try:
        return action.fcurves
    except AttributeError:
        return None


def main():
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    bpy.ops.import_scene.fbx(filepath=args.char)
    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("[export] character has no armature")
    if arm.animation_data:
        arm.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)

    joints, data, frame_time = read_bvh(args.bvh)
    rots, pos, rest = bvh_fk(joints, data)
    F = data.shape[0]
    bones = arm.data.bones
    names = [j[0] for j in joints]
    common = [k for k, n in enumerate(names) if n in bones]
    if len(common) < 3:
        raise SystemExit(f"[export] BVH bones do not match the character ({len(common)} in common)")

    # BVH space -> armature space
    src = rest[common]
    dst = np.array([list(bones[names[k]].head_local) for k in common])
    R, s, t = similarity(src, dst)
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    scale_ref = np.linalg.norm(dst - dst.mean(0), axis=1).mean() + 1e-9
    print(f"[export] {len(common)}/{len(names)} bones matched, rest fit error {err.mean() / scale_ref:.4f} "
          f"(relative), scale {s:.4f}")

    jidx = {n: k for k, n in enumerate(names)}
    # target armature-space matrices per frame
    order = []                                   # parents before children
    def visit(b):
        order.append(b)
        for c in b.children:
            visit(c)
    for b in bones:
        if b.parent is None:
            visit(b)

    rest_m = {b.name: b.matrix_local.copy() for b in bones}
    rest_rot = {b.name: np.array(b.matrix_local.to_3x3()) for b in bones}
    Rt = R.T

    scene = bpy.context.scene
    scene.render.fps = args.fps
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, F

    arm.animation_data_create()
    action = bpy.data.actions.new(name="MocapAnything")
    arm.animation_data.action = action
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    keys = {pb.name: ([], []) for pb in arm.pose.bones}    # name -> (quats, locs)
    for f in range(F):
        P = {}
        for b in order:
            k = jidx.get(b.name)
            if k is not None:
                rot = R @ rots[f, k] @ Rt @ rest_rot[b.name]
                p = s * (R @ pos[f, k]) + t
                M = Matrix([list(rot[0]) + [p[0]], list(rot[1]) + [p[1]], list(rot[2]) + [p[2]], [0, 0, 0, 1]])
            elif b.parent is not None:
                M = P[b.parent.name] @ (rest_m[b.parent.name].inverted() @ rest_m[b.name])
            else:
                M = rest_m[b.name].copy()
            P[b.name] = M
            if b.parent is not None:
                rel = rest_m[b.parent.name].inverted() @ rest_m[b.name]
                basis = rel.inverted() @ P[b.parent.name].inverted() @ M
            else:
                basis = rest_m[b.name].inverted() @ M
            loc, quat, _ = basis.decompose()
            if b.parent is not None and k is not None and jidx.get(b.parent.name) is not None:
                loc = Vector((0.0, 0.0, 0.0))     # bone lengths come from the rig, not the BVH
            q_list, l_list = keys[b.name]
            if q_list and Quaternion(q_list[-1]).dot(quat) < 0:   # keep quaternions continuous
                quat.negate()
            q_list.append(tuple(quat))
            l_list.append(tuple(loc))

    fcs = _fcurves(action)
    for pb in arm.pose.bones:
        q_list, l_list = keys[pb.name]
        for path, values, n in (("rotation_quaternion", q_list, 4), ("location", l_list, 3)):
            for i in range(n):
                dp = f'pose.bones["{pb.name}"].{path}'
                if fcs is not None:
                    fc = fcs.new(dp, index=i, action_group=pb.name)
                else:                                   # Blender 4.4+ slotted actions
                    fc = action.fcurve_ensure_for_datablock(arm, dp, index=i, group_name=pb.name)
                fc.keyframe_points.add(F)
                co = np.empty(2 * F)
                co[0::2] = np.arange(1, F + 1)
                co[1::2] = [v[i] for v in values]
                fc.keyframe_points.foreach_set("co", co)
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
                fc.update()
    # Blender 4.4+ (slotted actions): curves created through the legacy API live in a new slot that is
    # not assigned to the armature, so the action is not evaluated and the export bakes the rest pose.
    ad = arm.animation_data
    ad.action = action
    if hasattr(ad, "action_slot") and ad.action_slot is None and len(action.slots):
        ad.action_slot = action.slots[0]
    scene.frame_set(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    if args.format == "fbx":
        bpy.ops.export_scene.fbx(filepath=args.out, use_selection=False, add_leaf_bones=False,
                                 bake_anim=True, bake_anim_use_all_actions=False,
                                 bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True,
                                 bake_anim_simplify_factor=0.0)
    else:
        bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_animations=True,
                                  export_animation_mode="ACTIVE_ACTIONS", export_force_sampling=True)
    print(f"[export] {F} frames @ {args.fps} fps -> {args.out}")


main()
sys.stdout.flush()
