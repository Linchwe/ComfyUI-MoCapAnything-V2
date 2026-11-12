"""Clean-up passes on a MoCapAnything BVH (numpy only), applied before the FBX / GLB / BVH is saved.

MoCapAnything predicts the body pose relative to the hips; the hips stay at one fixed spot. These passes
work on that BVH:

  stabilize   "fix flips"   : single-frame pose errors (a joint snapping away and back) are replaced by
                              interpolation between the good frames around them
              "lock facing" : also removes the body's turning (rotation around the up axis), so the
                              character keeps facing one direction (front-facing dances, walks in place)
  mirror      copy the arm/leg motion of one side to the other side, mirrored (side views where the far
              side is hidden). Timing: "match original" lines the copy up with the beats (most forward /
              most backward leg) of the side it replaces; "half cycle" shifts it by half a step cycle;
              "same frame" copies without shift (jumps, symmetric moves)
  foot_lock   puts the lowest foot on the ground every frame and keeps the standing foot from sliding by
              moving the hips instead -> the character travels with its steps (root motion)
  in_place    keeps the up/down motion of foot_lock but removes travel on the ground

Axes are read from the character's rest pose: up = hips -> head, left = right leg -> left leg,
front = left x up. BVH joints have no rest rotation, so a local rotation R of a left bone mirrors to
M R M, M being the reflection across the body's middle plane.
"""
import math
import re

import numpy as np

STABILIZE = ["fix flips", "lock facing", "off"]
MIRROR_SIDES = ["off", "auto", "left -> right", "right -> left"]
MIRROR_TIMING = ["match original", "half cycle", "same frame"]


# ----------------------------------------------------------------------------- BVH I/O
class Bvh:
    def __init__(self, names, parents, offsets, channels, end_sites, data, frame_time):
        self.names, self.parents, self.offsets = names, parents, offsets
        self.channels, self.end_sites = channels, end_sites      # end_sites: {joint index: [offset, ...]}
        self.data, self.frame_time = data, frame_time


def read_bvh(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    head, motion = re.split(r"\bMOTION\b", text, maxsplit=1)
    tok = head.split()
    names, parents, offsets, channels, end_sites = [], [], [], [], {}
    stack, i, last, in_end = [], 0, None, False
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            last = len(names) - 1
            i += 2
        elif t == "End":
            in_end = True
            i += 2
        elif t == "{":
            stack.append(None if in_end else last)
            i += 1
        elif t == "}":
            if stack.pop() is None:
                in_end = False
            i += 1
        elif t == "OFFSET":
            v = np.array([float(x) for x in tok[i + 1:i + 4]])
            if in_end:
                end_sites.setdefault(stack[-2], []).append(v)
            else:
                offsets[last] = v
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            channels[last] = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    lines = [ln for ln in motion.strip().splitlines() if ln.strip()]
    n_frames = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    data = np.array([[float(x) for x in ln.split()] for ln in lines[2:2 + n_frames]], dtype=np.float64)
    return Bvh(names, parents, np.array(offsets), channels, end_sites, data, frame_time)


def _axis(a, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.zeros(r.shape + (3, 3))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[a]
    k = "XYZ".index(a)
    m[..., k, k] = 1
    m[..., i, i], m[..., i, j], m[..., j, i], m[..., j, j] = c, -s, s, c
    return m


def decode(b):
    """-> local rotations (F, J, 3, 3), root positions (F, 3)."""
    F, J = b.data.shape[0], len(b.names)
    L = np.tile(np.eye(3), (F, J, 1, 1))
    root = np.tile(b.offsets[0], (F, 1))
    col = 0
    for j, ch in enumerate(b.channels):
        for c in ch:
            v = b.data[:, col]
            if c.endswith("rotation"):
                L[:, j] = L[:, j] @ _axis(c[0].upper(), v)
            elif c.endswith("position") and j == 0:
                root[:, "XYZ".index(c[0].upper())] = v
            col += 1
    return L, root


def fk(b, L, root):
    """World rotations (F, J, 3, 3) and joint positions (F, J, 3)."""
    F, J = L.shape[:2]
    W = np.empty_like(L)
    P = np.empty((F, J, 3))
    for j, p in enumerate(b.parents):
        if p < 0:
            W[:, j], P[:, j] = L[:, j], root
        else:
            W[:, j] = W[:, p] @ L[:, j]
            P[:, j] = P[:, p] + np.einsum("fij,j->fi", W[:, p], b.offsets[j])
    return W, P


def _to_zyx(R):
    """R = Rz(a) Ry(b) Rx(c) -> degrees (a, b, c), unwrapped over time."""
    b = -np.arcsin(np.clip(R[..., 2, 0], -1, 1))
    a = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    c = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    out = np.degrees(np.stack([a, b, c], -1))
    return np.degrees(np.unwrap(np.radians(out), axis=0))


def write_bvh(path, b, L, root):
    """Write with standard channels (root: X/Y/Z position + Z/Y/X rotation, joints: Z/Y/X rotation)."""
    J = len(b.names)
    kids = {j: [k for k in range(J) if b.parents[k] == j] for j in range(J)}
    out = ["HIERARCHY"]

    def emit(j, depth):
        ind = "\t" * depth
        out.append(f"{ind}{'ROOT' if b.parents[j] < 0 else 'JOINT'} {b.names[j]}")
        out.append(ind + "{")
        o = b.offsets[j]
        out.append(f"{ind}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        if b.parents[j] < 0:
            out.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            out.append(f"{ind}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        for k in kids[j]:
            emit(k, depth + 1)
        for e in b.end_sites.get(j, []):
            out.extend([f"{ind}\tEnd Site", ind + "\t{", f"{ind}\t\tOFFSET {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}",
                        ind + "\t}"])
        out.append(ind + "}")

    emit(0, 0)
    F = L.shape[0]
    eul = _to_zyx(L)                                       # (F, J, 3)
    rows = np.concatenate([root, eul.reshape(F, -1)], 1)
    out += ["MOTION", f"Frames: {F}", f"Frame Time: {b.frame_time:.7f}"]
    out += [" ".join(f"{v:.6f}" for v in r) for r in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- rotation helpers
def mat2quat(R):
    R = np.asarray(R)
    q = np.empty(R.shape[:-2] + (4,))
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cand = np.stack([tr, R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], -1)
    k = cand.argmax(-1)
    for idx in range(4):
        m = k == idx
        if not m.any():
            continue
        r = R[m]
        if idx == 0:
            s = np.sqrt(1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]) * 2
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], -1)
        elif idx == 1:
            s = np.sqrt(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], -1)
        elif idx == 2:
            s = np.sqrt(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], -1)
        else:
            s = np.sqrt(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat2mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _continuous(q):
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def slerp(a, b, u):
    d = np.dot(a, b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + u * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def sample(R, tau):
    """Rotations R (F, 3, 3) sampled at fractional frames tau (clamped) with slerp."""
    F = len(R)
    q = _continuous(mat2quat(R))
    tau = np.clip(tau, 0, F - 1)
    out = np.empty((len(tau), 4))
    for i, t in enumerate(tau):
        a = int(math.floor(t))
        b = min(a + 1, F - 1)
        out[i] = slerp(q[a], q[b], t - a)
    return quat2mat(out)


def _smooth(x, k):
    if k <= 1 or len(x) < 3:
        return x
    k = min(k, len(x) - (1 - len(x) % 2))
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    w = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(xp, w, mode="valid")
    return np.stack([np.convolve(xp[:, i], w, mode="valid") for i in range(x.shape[1])], 1)


# ----------------------------------------------------------------------------- passes
def fix_flips(L, thresh_deg=45.0, win=7):
    """Replace frames where a joint's rotation jumps away from its neighbourhood (and back)."""
    F, J = L.shape[:2]
    fixed = 0
    if F < 5:
        return L, 0
    for j in range(J):
        q = _continuous(mat2quat(L[:, j]))
        pad = win // 2
        qp = np.concatenate([np.repeat(q[:1], pad, 0), q, np.repeat(q[-1:], pad, 0)], 0)
        med = np.stack([np.median(qp[t:t + win], 0) for t in range(F)])
        med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-12
        dev = np.degrees(2 * np.arccos(np.clip(np.abs((q * med).sum(1)), 0, 1)))
        bad = dev > thresh_deg
        if not bad.any() or bad.all():
            continue
        good = np.flatnonzero(~bad)
        for t in np.flatnonzero(bad):
            lo, hi = good[good < t], good[good > t]
            if len(lo) and len(hi):
                a, b = lo[-1], hi[0]
                q[t] = slerp(q[a], q[b], (t - a) / (b - a))
            else:
                q[t] = q[lo[-1] if len(lo) else hi[0]]
        L[:, j] = quat2mat(q)
        fixed += int(bad.sum())
    return L, fixed


def axes(b):
    """Body axes from the rest pose: up (hips -> head), left (right hip -> left hip), front = left x up."""
    J = len(b.names)
    _, P = fk(b, np.tile(np.eye(3), (1, J, 1, 1)), b.offsets[0][None])
    P = P[0]
    head = next((i for i, n in enumerate(b.names) if "head" in n.lower() and "top" not in n.lower()), None)
    up = (P[head] if head is not None else P[P[:, 1].argmax()]) - P[0]
    k = int(np.abs(up).argmax())                   # snap to the world axis (the head may lean forward)
    up = np.eye(3)[k] * np.sign(up[k])
    hl, hr = _leg(b.names, "L")[0], _leg(b.names, "R")[0]
    if hl is None or hr is None:
        pairs = side_pairs(b.names)
        if not pairs:
            return None
        hl, hr = pairs[0]
    left = P[hl] - P[hr]
    left = left - up * left.dot(up)
    k = int(np.abs(left).argmax())
    left = np.eye(3)[k] * np.sign(left[k])
    front = np.cross(left, up)
    return {"up": up, "left": left, "front": front, "M": np.eye(3) - 2 * np.outer(left, left), "rest": P}


def _rot_about(axis, a):
    a = np.asarray(a, float)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(a)[..., None, None] * K
            + (1 - np.cos(a))[..., None, None] * (K @ K))


def _yaw(R, ax):
    f = R @ ax["front"]                   # where the body's front points
    return np.arctan2(f @ ax["left"], f @ ax["front"])


def _body_yaw(b, L, root, ax):
    """Facing of the whole body per frame (radians): from the left-right line of the hips and the upper
    arms, so a pelvis that is predicted turned a few degrees against the chest does not set the facing."""
    pairs = []
    for words, excl in ((("upleg", "thigh", "upperleg"), ("twist",)),
                        (("arm",), ("fore", "lower", "twist", "hand", "_end"))):
        l, r = _find(b.names, "L", words, excl), _find(b.names, "R", words, excl)
        if l is not None and r is not None:
            pairs.append((l, r))
    if not pairs:
        return None
    _, P = fk(b, L, root)
    c = sum(P[:, l] - P[:, r] for l, r in pairs)
    c0 = sum(ax["rest"][l] - ax["rest"][r] for l, r in pairs)
    a = np.arctan2(-(c @ ax["front"]), c @ ax["left"])
    a0 = math.atan2(-(c0 @ ax["front"]), c0 @ ax["left"])
    return np.unwrap(a - a0)


def lock_facing(b, L, root, ax):
    """Remove the body's turning around the up axis: the character always faces its own front, so a walk
    goes forward and a dance faces the front, which is what game animation clips expect."""
    yaw = _body_yaw(b, L, root, ax)
    if yaw is None:
        yaw = np.unwrap(_yaw(L[:, 0], ax))
    L[:, 0] = _rot_about(ax["up"], -yaw) @ L[:, 0]
    return L, float(np.degrees(yaw.max() - yaw.min()))


# --- left / right pairs
_SIDE = [("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT")]
_SIDE_RE = [(r"(^|[_.:\s-])L([_.:\s-]|$)", r"(^|[_.:\s-])R([_.:\s-]|$)"),
            (r"([_.:\s-])l$", r"([_.:\s-])r$")]


def _side_of(name):
    for a, b in _SIDE:
        if a in name:
            return "L", name.replace(a, b)
        if b in name:
            return "R", name.replace(b, a)
    for ra, rb in _SIDE_RE:
        if re.search(ra, name):
            return "L", re.sub(ra, lambda m: m.group(0).replace("L", "R").replace("l", "r"), name)
        if re.search(rb, name):
            return "R", re.sub(rb, lambda m: m.group(0).replace("R", "L").replace("r", "l"), name)
    return None, None


def side_pairs(names):
    idx = {n: i for i, n in enumerate(names)}
    pairs = []                                  # (left index, right index)
    for i, n in enumerate(names):
        s, other = _side_of(n)
        if s == "L" and other in idx:
            pairs.append((i, idx[other]))
    return pairs


def _find(names, side, words, exclude=()):
    for i, n in enumerate(names):
        s, _ = _side_of(n)
        low = n.lower()
        if s == side and any(w in low for w in words) and not any(x in low for x in exclude):
            return i
    return None


def _leg(names, side):
    hip = _find(names, side, ("upleg", "thigh", "upperleg", "hip"), ("twist",))
    knee = _find(names, side, ("leg", "calf", "knee", "shin"), ("upleg", "upper", "twist"))
    foot = _find(names, side, ("foot", "ankle"), ("twist",))
    toe = _find(names, side, ("toe",), ("end", "_end"))
    return hip, knee, foot, toe


def _leg_signal(b, L, side, ax):
    """Forward swing of the thigh (degrees) in the hips' frame, per frame."""
    hip, knee, _, _ = _leg(b.names, side)
    if hip is None or knee is None:
        return None
    L2 = L.copy()
    L2[:, 0] = np.eye(3)                          # ignore the body's turning
    _, P = fk(b, L2, np.zeros((L.shape[0], 3)))
    v = P[:, knee] - P[:, hip]
    return np.degrees(np.arctan2(v @ ax["front"], -(v @ ax["up"])))


def _period(sig):
    x = sig - sig.mean()
    if np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lo = 8                                         # > 0.25 s at 30 fps
    if len(ac) <= lo + 2:
        return None
    k = lo + int(np.argmax(ac[lo:len(ac) // 2 + 1])) if len(ac) // 2 + 1 > lo else None
    if k is None or ac[k] < 0.3:
        return None
    return float(k)


def _events(sig, period):
    """Frames of the leg's most-forward and most-backward points (local extremes)."""
    s = _smooth(sig, 5)
    d = max(3, int((period or 20) * 0.35))
    prom = 0.25 * (s.std() + 1e-9)
    ev = []
    for sign in (1, -1):
        x = sign * s
        for t in range(1, len(x) - 1):
            lo, hi = max(0, t - d), min(len(x), t + d + 1)
            if x[t] == x[lo:hi].max() and x[t] - np.median(x[lo:hi]) > prom:
                ev.append((t, sign))
    ev.sort()
    return ev


def _time_map(src_sig, dst_sig, F, mode, log):
    """tau[t] = frame of the source side used for frame t of the replaced side."""
    t = np.arange(F, dtype=float)
    if mode == "same frame" or src_sig is None:
        return t, "same frame"
    T = _period(src_sig) or (_period(dst_sig) if dst_sig is not None else None)
    if T is None:
        log("[motion] mirror: no step cycle found, copying on the same frame")
        return t, "same frame"

    def half():
        tau = t - T / 2
        tau[tau < 0] += T
        return tau

    if mode == "half cycle" or dst_sig is None:
        return half(), f"half cycle ({T / 30:.2f} s cycle)"
    es, ed = _events(src_sig, T), _events(dst_sig, T)
    # mirrored source sign == destination sign marks the same beat (forward/backward swing)
    # each beat of the replaced side takes the latest earlier beat of the same kind on the source side
    # (about half a cycle before), so the map runs one way and stays increasing
    pairs = []
    for td, sd in ed:
        cand = [ts for ts, ss in es if ss == sd and 0.2 * T <= td - ts <= 0.8 * T]
        if cand:
            pairs.append((td, max(cand)))
    pairs = sorted(dict(pairs).items())
    clean = []
    for td, ts in pairs:                                      # keep the map increasing
        if not clean or (td > clean[-1][0] and ts > clean[-1][1]):
            clean.append((td, ts))
    if len(clean) < 2:
        log("[motion] mirror: beats of the replaced side not found, using half cycle")
        return half(), f"half cycle ({T / 30:.2f} s cycle)"
    xd, xs = np.array([c[0] for c in clean], float), np.array([c[1] for c in clean], float)
    tau = np.interp(t, xd, xs)
    tau[t < xd[0]] = xs[0] - (xd[0] - t[t < xd[0]])           # before the first beat: same pace
    tau[t > xd[-1]] = xs[-1] + (t[t > xd[-1]] - xd[-1])
    tau[tau < 0] += T
    tau[tau > F - 1] -= T
    return tau, f"match original ({len(clean)} beats)"


def _jitter_score(L, idx):
    s = 0.0
    for j in idx:
        q = _continuous(mat2quat(L[:, j]))
        acc = np.linalg.norm(q[2:] - 2 * q[1:-1] + q[:-2], axis=1).mean() if len(q) > 2 else 0
        amp = np.linalg.norm(q - q.mean(0), axis=1).mean() + 1e-6
        s += acc / amp
    return s / max(1, len(idx))


def mirror(b, L, side, timing, ax, log=print):
    pairs = side_pairs(b.names)
    if not pairs:
        log("[motion] mirror: no Left/Right bone pairs found - skipped")
        return L, None
    lefts, rights = [p[0] for p in pairs], [p[1] for p in pairs]
    if side == "auto":
        sl, sr = _jitter_score(L, lefts), _jitter_score(L, rights)
        side = "left -> right" if sl <= sr else "right -> left"
        log(f"[motion] mirror auto: steadier side = {'left' if sl <= sr else 'right'}")
    src_is_left = side.startswith("left")
    src_sig = _leg_signal(b, L, "L" if src_is_left else "R", ax)
    dst_sig = _leg_signal(b, L, "R" if src_is_left else "L", ax)
    F = L.shape[0]
    tau, how = _time_map(src_sig, dst_sig, F, timing, log)
    out = L.copy()
    for li, ri in pairs:
        s, d = (li, ri) if src_is_left else (ri, li)
        out[:, d] = ax["M"] @ sample(L[:, s], tau) @ ax["M"]
    log(f"[motion] mirror {side}: {len(pairs)} bone pairs, timing = {how}")
    return out, side


def foot_lock(b, L, root, ax, in_place=False, log=print):
    """Ground the lowest foot every frame; keep the standing foot still by moving the hips."""
    F = L.shape[0]
    feet = {}
    for s in ("L", "R"):
        _, _, foot, toe = _leg(b.names, s)
        if foot is None:
            log("[motion] foot lock: foot bones not found - skipped")
            return root
        feet[s] = [foot] + ([toe] if toe is not None else [])
    up = ax["up"]
    Pr = ax["rest"]
    ground = min(Pr[j] @ up for s in feet for j in feet[s])
    leg_len = float(np.linalg.norm(Pr[0] - Pr[feet["L"][0]])) or 1.0
    hyst = 0.02 * leg_len

    _, P = fk(b, L, root)
    H = P @ up                                              # heights (F, J)
    flat = P - H[..., None] * up                            # positions on the ground plane
    low = {s: H[:, feet[s]].min(1) for s in feet}
    dy = _smooth(ground - np.minimum(low["L"], low["R"]), 3)
    off = np.zeros((F, 3))
    anchor = "L" if low["L"][0] <= low["R"][0] else "R"
    lock = flat[0, feet[anchor][0]].copy()
    for t in range(F):
        other = "R" if anchor == "L" else "L"
        if t and low[other][t] < low[anchor][t] - hyst:     # weight moved to the other foot
            anchor = other
            lock = flat[t, feet[anchor][0]] + off[t - 1]
        off[t] = lock - flat[t, feet[anchor][0]]
    travel = float(np.linalg.norm(off[-1] - off[0]))
    if in_place:
        off[:] = 0.0
    off += dy[:, None] * up
    log("[motion] foot lock: lowest foot on the ground"
        + (", in place" if in_place else f", travel {travel / leg_len:.2f} leg lengths"))
    return root + off


def process(src, dst, stabilize="fix flips", mirror_side="off", mirror_timing="match original",
            foot_lock_on=False, in_place=False, log=print):
    """Apply the passes; returns True when dst was written (something changed)."""
    if stabilize == "off" and mirror_side == "off" and not foot_lock_on and not in_place:
        return False
    b = read_bvh(src)
    L, root = decode(b)
    ax = axes(b)
    if ax is None and (stabilize == "lock facing" or mirror_side != "off" or foot_lock_on):
        log("[motion] left/right legs not found in the skeleton - only 'fix flips' / 'in place' are applied")
    if stabilize in ("fix flips", "lock facing"):
        L, n = fix_flips(L)
        log(f"[motion] fix flips: {n} joint-frames repaired")
    if stabilize == "lock facing" and ax is not None:
        L, turn = lock_facing(b, L, root, ax)
        log(f"[motion] lock facing: removed up to {turn:.0f} deg of turning")
    if mirror_side != "off" and ax is not None:
        L, _ = mirror(b, L, mirror_side, mirror_timing, ax, log)
    if foot_lock_on and ax is not None:
        root = foot_lock(b, L, root, ax, in_place, log)
    elif in_place:
        root = np.tile(root[:1], (len(root), 1))
    write_bvh(dst, b, L, root)
    return True
