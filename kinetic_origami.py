"""Kinetic origami garments, linkage-driven ribs. Blender 5.2 LTS (bpy).
Design A "bud":    8 radial ribs, one big pleated cloth, closes like an umbrella into a pod.
Design B "spiral": same rib modules on skewed (pinwheel) planes, spiral pleats, twisted bud, spins when closed.
build:  python kinetic_origami.py build A|B OUT.blend
render: python kinetic_origami.py render IN.blend CAM OUTDIR PCT FRAMES   (FRAMES "25-300" or "40,120")
"""
import bpy, sys, math, os
import numpy as np
from mathutils import Vector, Matrix, Quaternion

# ---------------- parameters (metres, seconds, degrees) ----------------
FPS, PRE = 24, 24
WAIST_Z, HUB_X, HUB_Y = 1.00, 0.05, -0.178     # hub centre; wearer faces -Y, hub 5 cm to wearer's left
N_RIB, R0 = 8, 0.045                           # rib count, crown pivot radius
L = (0.060, 0.057, 0.053)                      # rib segment lengths (pin to pin)
SEG_W, SEG_H = 0.006, 0.005                    # rib bar width (along pin) and height
PIN_R, KNUCKLE_R = 0.002, 0.0032               # M4 shoulder pins, joint knuckle radius
CA = (0.008, -0.004); CB = (0.004, 0.008)      # crossed four-bar pins: on input body / output body, from joint
TAIL, ROD = 0.012, 0.020                       # J1 lever behind pivot, pushrod length to runner
SCREW_LEAD, MOTOR_RPM = 0.0005, 600            # M3 x 0.5 lead screw, N20 gearmotor speed under load
ACC_T = 0.25                                   # motor ramp time
TIP_GAP = 0.016                                # closed: rib tip centreline distance from the axis plane (A)
STAND = 0.010                                  # fabric held this far behind rib centreline (convex side)
FAB_IN, FAB_R, LOBE = 0.044, 0.222, 0.014      # fabric clamp radius (back-plate rim), radius at rib, scallop bulge
SKIRT_R, PLATE_Y = 0.056, -0.044               # fabric rises from the back-plate rim to the rib plane by SKIRT_R
NT, NRAD = 160, 30                             # fabric columns / rows
PLEAT_K = 0.07                                 # pleat depth per metre of radius (open, heat-set)
GSM, MASS_SCALE = 120.0, 1000.0                # fabric g/m2; masses and forces scaled together (dynamics unchanged)
BEND, BEND_CREASE = 60.0, 6.0                  # calibrated: 4.0 cm Peirce bending length at 6 mm mesh
DESIGNS = dict(
    A=dict(beta=0.0, spin_turns=0.0, t_close=2.5, t_hold=2.5, t_end=2.5),
    B=dict(beta=25.0, spin_turns=2.0, t_close=2.5, t_hold=5.6, t_end=2.0, spin_rpm=30.0, spin_ramp=0.6))

C = Vector((HUB_X, HUB_Y, WAIST_Z)); W = Vector((0, -1, 0))   # W: outward axis
def rad(th): return Vector((math.cos(th), 0, math.sin(th)))
R2 = lambda a: np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

# ---------------- kinematics (planar, per rib) ----------------
def four_bar(Lf, qin, guess):
    """frame = middle link (J at 0, next J at Lf); input body turns -qin about 0; solve output angle."""
    A = R2(-qin) @ np.array(CA); c = np.linalg.norm(np.array([Lf, 0]) + CB - np.array(CA))
    D = A - np.array([Lf, 0]); rb, d = np.hypot(*CB), np.linalg.norm(D)
    ca = np.clip((rb * rb + d * d - c * c) / (2 * rb * d), -1, 1)
    s = [math.atan2(D[1], D[0]) + k * math.acos(ca) - math.atan2(CB[1], CB[0]) for k in (1, -1)]
    return min([(x + math.pi) % (2 * math.pi) - math.pi for x in s], key=lambda x: abs(x - guess))

def rib_pose(q1):
    """joint angles, joint positions (2D, origin J1, x along rib, y outward axis)"""
    q2 = four_bar(L[0], q1, q1); q3 = four_bar(L[1], q2, q2)
    J, phi = [np.zeros(2)], [q1, q1 + q2, q1 + q2 + q3]
    for p, l in zip(phi, L): J.append(J[-1] + R2(p) @ np.array([l, 0]))
    return (q1, q2, q3), phi, J

def runner_s(q):  # runner travel toward body for J1 angle q (tail lever + pushrod)
    return e_sin(q) + math.sqrt(ROD ** 2 - (TAIL * (1 - math.cos(q))) ** 2) - ROD
def e_sin(q): return TAIL * math.sin(q)

def closed_q1(rho):
    """J1 angle where the tip reaches TIP_GAP from the axis (measured in the rib plane)"""
    s0 = math.sqrt(R0 ** 2 - rho ** 2)
    for q in np.arange(0.5, 1.4, 0.0005):
        if s0 + rib_pose(q)[2][3][0] <= TIP_GAP: return q
    raise RuntimeError("rib never closes")

def ease_profile(dist, vmax, acc_t):
    """trapezoidal velocity profile: returns time span and position(t)"""
    a = vmax / acc_t
    if dist < vmax * acc_t: acc_t = math.sqrt(dist / a); vmax = a * acc_t
    T = dist / vmax + acc_t
    def x(t):
        t = min(max(t, 0), T)
        if t < acc_t: return 0.5 * a * t * t
        if t > T - acc_t: return dist - 0.5 * a * (T - t) ** 2
        return 0.5 * a * acc_t ** 2 + vmax * (t - acc_t)
    return T, x

# ---------------- mesh helpers ----------------
def mat(name, col, rough=0.5, metal=0.0, sheen=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal; b.inputs["Sheen Weight"].default_value = sheen
    return m

class MB:  # mesh builder in a local frame
    def __init__(s): s.v, s.f = [], []
    def add(s, v, f, M=Matrix()):
        o = len(s.v); s.v += [tuple(M @ Vector(p)) for p in v]; s.f += [tuple(i + o for i in q) for q in f]; return s
    def box(s, x0, x1, y0, y1, z0, z1, M=Matrix()):
        v = [(x1 if i & 1 else x0, y1 if i & 2 else y0, z1 if i & 4 else z0) for i in range(8)]
        return s.add(v, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)], M)
    def cyl(s, r, z0, z1, M=Matrix(), n=16, rin=0.0):  # along local Z
        v = [(rr * math.cos(2 * math.pi * i / n), rr * math.sin(2 * math.pi * i / n), z) for z in (z0, z1) for rr in ((rin, r) if rin else (r,)) for i in range(n)]
        if not rin:
            f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        else:
            q = lambda a, b, i: (a * n + i, a * n + (i + 1) % n, b * n + (i + 1) % n, b * n + i)
            f = sum([[q(1, 3, i), q(3, 2, i), q(2, 0, i), q(0, 1, i)] for i in range(n)], [])
        return s.add(v, f, M)
    def obj(s, name, m, col=None):
        me = bpy.data.meshes.new(name); me.from_pydata(s.v, [], s.f); me.update()
        o = bpy.data.objects.new(name, me); o.data.materials.append(m); (col or bpy.context.scene.collection).objects.link(o)
        return o

def frame(origin, x, y):  # columns x, y, z=x*y
    x = x.normalized(); y = (y - y.dot(x) * x).normalized(); z = x.cross(y)
    M = Matrix((x, y, z)).transposed().to_4x4(); M.translation = origin; return M
def M2(p, a): return Matrix.Translation((p[0], p[1], 0)) @ Matrix.Rotation(a, 4, 'Z')

# ---------------- build ----------------
def build(design, path):
    D = DESIGNS[design]; beta = math.radians(D["beta"]); rho = R0 * math.sin(beta)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    sc = bpy.context.scene
    M_AL = mat("anodised_aluminium", (0.05, 0.052, 0.056), 0.35, 1.0)
    M_ST = mat("steel", (0.55, 0.56, 0.58), 0.25, 1.0)
    M_HUB = mat("hub_nylon", (0.03, 0.03, 0.032), 0.6)
    M_FAB = mat("fabric", (0.80, 0.78, 0.73), 0.8, sheen=0.4)
    M_TORSO = mat("torso", (0.13, 0.125, 0.12), 0.7)
    M_MOTOR = mat("motor_brass", (0.6, 0.45, 0.2), 0.3, 1.0)
    M_TEXT = mat("touch_text", (1, 1, 1), 1.0)
    M_TEXT.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 1, 1, 1)
    M_TEXT.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.5

    # torso proxy (same as lotus)
    secs = [(0.70, 0.165, 0.11), (0.80, 0.19, 0.12), (0.90, 0.18, 0.112), (WAIST_Z, 0.17, 0.105), (1.10, 0.168, 0.108),
            (1.20, 0.175, 0.12), (1.28, 0.185, 0.135), (1.36, 0.195, 0.12), (1.43, 0.20, 0.10), (1.475, 0.15, 0.08),
            (1.50, 0.065, 0.055), (1.58, 0.058, 0.05)]
    n, ex = 64, 2.6; tb = MB(); v = []
    for z, a, b in secs:
        for i in range(n):
            c, s = math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)
            v.append((a * math.copysign(abs(c) ** (2 / ex), c), b * math.copysign(abs(s) ** (2 / ex), s), z))
    f = [(j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i) for j in range(len(secs) - 1) for i in range(n)]
    v += [(0, 0, secs[0][0]), (0, 0, secs[-1][0])]
    f += [(len(v) - 2, (i + 1) % n, i) for i in range(n)] + [(len(v) - 1, (len(secs) - 1) * n + i, (len(secs) - 1) * n + (i + 1) % n) for i in range(n)]
    torso = tb.add(v, f).obj("Torso", M_TORSO); torso.data.shade_smooth()
    torso.modifiers.new("Subsurf", 'SUBSURF').levels = 2
    torso.modifiers.new("Collision", 'COLLISION'); torso.collision.thickness_outer = 0.003

    AXW = frame(C, Vector((1, 0, 0)), Vector((0, 0, 1)))  # local Z = W (outward); disc in local XY
    AX = AXW @ Matrix.Rotation(math.pi, 4, 'X')            # local Z = -W (into body)
    body_y = -0.075                                       # torso surface relative to rib plane, along W
    # static base: belt-mounted housing, fold motor (A) or bearing + spin motor (B)
    base = MB().cyl(0.050, -body_y - 0.004, 0.047, AX, 48)
    if design == "B":
        base.cyl(0.050, 0.044, 0.047, AX, 48, rin=0.042)                      # 6806 bearing outer ring 42 mm OD (visual)
        base.box(-0.006, 0.006, -0.005, 0.005, 0.047, 0.060, AX @ Matrix.Translation((0.044, 0, 0)))  # N20 spin motor, offset
    base.obj("Base", M_HUB)
    # rotating carrier: front crown ring, back plate, guide rods, lead screw, fold motor (N20, 12x10 mm)
    car = MB().cyl(0.041, 0.0035, 0.0065, AXW, 48, rin=0.022).cyl(FAB_IN, PLATE_Y - 0.003, PLATE_Y, AXW, 48)
    for j in range(3):
        th = 2 * math.pi * j / 3 + 0.3
        car.cyl(0.0015, PLATE_Y, 0.0035, AXW @ Matrix.Translation((0.018 * math.cos(th), 0.018 * math.sin(th), 0)), 10)
    car.cyl(0.0015, PLATE_Y, 0.0, AXW, 10)
    car_mot = MB().box(-0.006, 0.006, -0.005, 0.005, -0.071, -0.047, AXW)   # N20 lead-screw motor, inside the base
    if design == "B": car_mot.cyl(0.019, -0.060, -0.054, AXW, 24, rin=0.012)  # through-bore slip ring (visual)

    # rib frames
    def col_point(j, r):  # fabric column j at radius r (tangent lines to circle rho; radial when rho = 0)
        g = (lambda r: math.atan2(math.sqrt(max(r * r - rho * rho, 0)), rho)) if rho > 0 else (lambda r: 0.0)
        return C + rad(2 * math.pi * j / NT + g(r) - g(R0)) * r
    RF = []
    for k in range(N_RIB):
        j = k * NT // N_RIB; P0 = col_point(j, R0); nvec = (col_point(j, R0 + 1e-4) - P0).normalized()
        RF.append(frame(P0, nvec, W))
    # crown clevis cheeks, crossed four-bar ground pins, runner pivots, all in rib frames
    q_cl = closed_q1(rho); S_MAX = runner_s(q_cl)
    for k in range(N_RIB):
        F = RF[k]
        for zs in (-1, 1):
            car.box(-0.004, 0.005, -0.006, 0.0065, zs * 0.0062 - 0.0009, zs * 0.0062 + 0.0009, F)   # cheeks hold J1, hang from ring
            car.box(-0.004, CA[0] + 0.003, CA[1] - 0.003, CA[1] + 0.003, zs * 0.0074 - 0.0006, zs * 0.0074 + 0.0006, F)
        car.cyl(PIN_R, -0.0075, 0.0075, F, 12).cyl(PIN_R * 0.8, -0.0062, 0.0062, F @ Matrix.Translation((*CA, 0)), 10)
    carrier = car.obj("Carrier", M_HUB); motor = car_mot.obj("FoldMotor", M_MOTOR)
    motor.parent = carrier
    carrier.modifiers.new("Collision", 'COLLISION').settings.thickness_outer = 0.002

    # rib segments (bars + knuckles + pins + fabric posts), couplers (paired side plates), pushrods, runner
    parts = {}
    for k in range(N_RIB):
        for i in range(3):
            b = MB(); x0 = -TAIL - 0.003 if i == 0 else 0.0
            b.box(x0, L[i], -SEG_H / 2, SEG_H / 2, -SEG_W / 2, SEG_W / 2)
            b.cyl(KNUCKLE_R, -SEG_W / 2, SEG_W / 2, n=16)
            if i < 2: b.cyl(PIN_R, -0.0075, 0.0075, Matrix.Translation((L[i], 0, 0)), 12)
            if i == 2: b.cyl(KNUCKLE_R, -SEG_W / 2, SEG_W / 2, Matrix.Translation((L[i], 0, 0)), 16)
            if i == 0: b.cyl(PIN_R * 0.8, -0.0062, 0.0062, Matrix.Translation((-TAIL, 0, 0)), 10)
            if i < 2:   # input crank boss for next four-bar (A pin)
                b.box(L[i], L[i] + CA[0] + 0.003, CA[1] - 0.0028, 0, -SEG_W / 2, SEG_W / 2)
                b.cyl(PIN_R * 0.8, -0.0062, 0.0062, Matrix.Translation((L[i] + CA[0], CA[1], 0)), 10)
            if i > 0:   # output crank boss (B pin)
                b.box(CB[0] - 0.003, CB[0] + 0.003, 0, CB[1] + 0.0028, -SEG_W / 2, SEG_W / 2)
            for x in np.arange(0.012 if i == 0 else 0.008, L[i] - 0.004, 0.012):   # fabric posts on convex side
                b.cyl(0.0012, SEG_H / 2, STAND - 0.0012, Matrix.Translation((x, 0, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'X'), 8)
            o = b.obj(f"Rib{k}_Seg{i}", M_AL); parts[o.name] = o
            o.modifiers.new("Collision", 'COLLISION').settings.thickness_outer = 0.0015
        for i in range(2):
            Lc = float(np.linalg.norm(np.array([L[i], 0]) + CB - np.array(CA)))
            b = MB()
            for zs in (-1, 1):
                z0 = zs * (0.0045 if i == 1 else 0.0062)
                b.box(-0.003, Lc + 0.003, -0.0025, 0.0025, z0 - 0.0006, z0 + 0.0006)
            b.cyl(PIN_R * 0.8, -0.0068, 0.0068, n=10).cyl(PIN_R * 0.8, -0.0068, 0.0068, Matrix.Translation((Lc, 0, 0)), 10)
            parts[f"Rib{k}_Link{i}"] = b.obj(f"Rib{k}_Link{i}", M_ST)
        b = MB().cyl(0.0012, 0, ROD, Matrix.Rotation(math.pi / 2, 4, 'Y'), 8)
        for x in (0, ROD): b.cyl(0.0028, -0.002, 0.002, Matrix.Translation((x, 0, 0)), 12)
        parts[f"Rib{k}_Rod"] = b.obj(f"Rib{k}_Rod", M_ST)
    rb = MB().cyl(R0 - TAIL + 0.004, -ROD - 0.006, -ROD - 0.002, AXW, 48, rin=0.007)
    for k in range(N_RIB):  # runner clevis for each pushrod
        rb.box(-TAIL - 0.003, -TAIL + 0.003, -ROD - 0.006, -ROD + 0.0, -0.004, 0.004, RF[k])
    runner = rb.obj("Runner", M_HUB)

    # ---------- motion ----------
    v_lin = SCREW_LEAD * MOTOR_RPM / 60
    T_mv, s_of = ease_profile(S_MAX, v_lin, ACC_T)
    t0 = D["t_close"]; t1 = t0 + T_mv; t2 = t1 + D["t_hold"]; t3 = t2 + T_mv; DUR = t3 + D["t_end"]
    sc.render.fps, sc.frame_start, sc.frame_end = FPS, PRE + 1, PRE + int(round(DUR * FPS))
    if D["spin_turns"]:
        w_max = D["spin_rpm"] / 60 * 2 * math.pi; T_sp, psi_of = ease_profile(D["spin_turns"] * 2 * math.pi, w_max, D["spin_ramp"])
        ts0 = t1 + (D["t_hold"] - T_sp) / 2
    qs = np.linspace(0, q_cl + 0.02, 600); ss = np.array([runner_s(q) for q in qs])
    def state(t):
        s = s_of(t - t0) if t < t2 else S_MAX - s_of(t - t2)
        psi = psi_of(t - ts0) if D["spin_turns"] else 0.0
        return s, float(np.interp(s, ss, qs)), psi
    log = []
    for fr in range(1, sc.frame_end + 1):
        t = (fr - PRE - 1) / FPS; s, q1, psi = state(t)
        SP = Matrix.Translation(C) @ Matrix.Rotation(psi, 4, W) @ Matrix.Translation(-C)
        mats = {"Carrier": SP, "Runner": SP @ Matrix.Translation(-W * s)}
        q, phi, J = rib_pose(q1)
        Tl = R2(q1) @ np.array([-TAIL, 0]); U = np.array([-TAIL, -ROD - s])
        for k in range(N_RIB):
            F = SP @ RF[k]
            for i in range(3): mats[f"Rib{k}_Seg{i}"] = F @ M2(J[i], phi[i])
            for i, (A_, P_, a_in, a_out) in enumerate(((np.array(CA), J[1], 0.0, phi[1]),
                                                        (J[1] + R2(phi[0]) @ np.array(CA), J[2], phi[0], phi[2]))):
                B_ = P_ + R2(a_out) @ np.array(CB)
                mats[f"Rib{k}_Link{i}"] = F @ M2(A_, math.atan2(*(B_ - A_)[::-1]))
            mats[f"Rib{k}_Rod"] = F @ M2(U, math.atan2(*(Tl - U)[::-1]))
        for nm, Mw in mats.items():
            o = parts.get(nm) or bpy.data.objects[nm]; o.rotation_mode = 'QUATERNION'
            qn = Mw.to_quaternion(); pq = o.get("_pq")
            if pq is not None and Quaternion(pq).dot(qn) < 0: qn.negate()
            o["_pq"] = list(qn); o.location, o.rotation_quaternion = Mw.translation, qn
            o.keyframe_insert("location", frame=fr); o.keyframe_insert("rotation_quaternion", frame=fr)
        log.append((t, s, *np.degrees(q), math.degrees(psi)))
    for o in bpy.data.objects:
        if "_pq" in o: del o["_pq"]
    sc.frame_set(1)

    # ---------- fabric: pleated, scalloped disc; columns follow rib lines ----------
    sec = NT // N_RIB; v, f, pin, crease, hook_of = [], [], [], [], []
    for j in range(NT):
        u = (j % sec) / sec; Rt = FAB_R + LOBE * math.sin(math.pi * u) ** 2
        tri = 1 - abs(((j % (sec // 2)) / (sec // 4)) - 1)       # 0 at ribs and mid-sector valley, 1 at the two fins
        for i in range(NRAD + 1):
            r = FAB_IN + (Rt - FAB_IN) * i / NRAD
            sk = min(max((SKIRT_R - r) / (SKIRT_R - FAB_IN), 0), 1)          # 1 at the clamp, 0 from SKIRT_R out
            p = col_point(j, r) - W * (STAND + (-PLATE_Y - STAND) * sk ** 1.5 + PLEAT_K * max(r - SKIRT_R, 0) * tri)
            v.append(p); h = None
            if i == 0: h = "Carrier"
            elif j % sec == 0:
                sr = math.sqrt(max(r * r - rho * rho, 0)) - math.sqrt(R0 * R0 - rho * rho)
                if 0.006 < sr <= sum(L) - 0.002:
                    ii = 0 if sr <= L[0] else (1 if sr <= L[0] + L[1] else 2)
                    h = f"Rib{(j // sec) % N_RIB}_Seg{ii}"
            hook_of.append(h); crease.append(j % (sec // 4) == 0 and j % sec != 0)
    for j in range(NT):
        a_, b_ = j * (NRAD + 1), ((j + 1) % NT) * (NRAD + 1)
        f += [(a_ + i, b_ + i, b_ + i + 1, a_ + i + 1) for i in range(NRAD)]
    fab = MB().add([tuple(p) for p in v], f).obj("Fabric", M_FAB); fab.data.shade_smooth()
    gp, gfree, gb = (fab.vertex_groups.new(name=nm) for nm in ("pin", "free", "bend"))
    for vi, (h, cr) in enumerate(zip(hook_of, crease)):
        (gp if h else gfree).add([vi], 1.0, 'REPLACE'); gb.add([vi], 0.0 if cr else 1.0, 'REPLACE')
    for nm in sorted(set(h for h in hook_of if h)):
        idx = [vi for vi, h in enumerate(hook_of) if h == nm]; tgt = bpy.data.objects[nm]
        hm = fab.modifiers.new(f"hook_{nm}", 'HOOK'); hm.object = tgt; hm.falloff_type = 'NONE'
        hm.matrix_inverse = tgt.matrix_world.inverted(); hm.vertex_indices_set(idx)
    area = sum(p.area for p in fab.data.polygons)
    cl = fab.modifiers.new("Cloth", 'CLOTH'); cs = cl.settings
    cs.quality, cs.mass, cs.air_damping = 10, GSM / 1000 * area / len(v) * MASS_SCALE, 0.1
    cs.bending_model = 'ANGULAR'
    cs.tension_stiffness = cs.compression_stiffness = 2000.0; cs.shear_stiffness = 200.0
    cs.bending_stiffness, cs.bending_stiffness_max, cs.vertex_group_bending = BEND_CREASE, BEND, "bend"
    cs.tension_damping = cs.compression_damping = cs.shear_damping = 10.0; cs.bending_damping = 2.0
    cs.vertex_group_mass, cs.pin_stiffness = "pin", 1.0
    cc = cl.collision_settings
    cc.collision_quality, cc.distance_min, cc.use_self_collision, cc.self_distance_min = 4, 0.0015, True, 0.0015
    cc.vertex_group_object_collisions = "free"
    cl.point_cache.frame_start, cl.point_cache.frame_end = 1, sc.frame_end
    fab.modifiers.new("Solidify", 'SOLIDIFY').thickness = 0.0005
    fab.modifiers.new("Subsurf", 'SUBSURF').levels = 1
    fab["fabric_mass_g"] = GSM * area

    # ---------- studio ----------
    w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.22, 0.22, 0.23, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    for nm, loc, size, en in (("Key", (1.6, -1.8, 2.2), 1.5, 110), ("Fill", (-1.8, -1.4, 1.3), 2.0, 35),
                              ("Rim", (0.6, 1.6, 2.0), 1.0, 70), ("Low", (0.4, -1.0, 0.2), 1.2, 12)):
        Lg = bpy.data.lights.new(nm, 'AREA'); Lg.size, Lg.energy = size, en
        lo = bpy.data.objects.new(nm, Lg); sc.collection.objects.link(lo); lo.location = loc
        lo.rotation_euler = (Vector((C.x, -0.2, 1.05)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    for nm, loc, tgt, lens in (("CamFront3Q", (1.05, -1.25, 1.22), (0.07, -0.26, 1.02), 70),
                               ("CamMech", (C.x + 0.30, C.y - 0.30, C.z + 0.17), (C.x + 0.06, C.y - 0.07, C.z + 0.02), 50)):
        cam = bpy.data.objects.new(nm, bpy.data.cameras.new(nm)); sc.collection.objects.link(cam)
        cam.data.lens, cam.data.clip_start = lens, 0.005; cam.location = loc
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        d = 0.05; hh = d * 18 / lens
        tx = bpy.data.objects.new(f"Touch_{nm}", bpy.data.curves.new(f"Touch_{nm}", 'FONT')); sc.collection.objects.link(tx)
        tx.data.body, tx.data.size, tx.data.align_x = "touch", 0.09 * hh, 'CENTER'
        tx.parent = cam; tx.location = (0, -0.82 * hh, -d); tx.data.materials.append(M_TEXT)
        for t, val in ((t0 - 0.15, 0), (t0, 1), (t0 + 1.2, 1), (t0 + 1.35, 0)):
            tx.scale = (val,) * 3; tx.keyframe_insert("scale", frame=PRE + 1 + t * FPS)
    sc.camera = bpy.data.objects["CamFront3Q"]
    r = sc.render; r.engine = 'BLENDER_EEVEE'; r.resolution_x, r.resolution_y = 1080, 1350
    sc.eevee.taa_render_samples = 10; sc.eevee.shadow_ray_count = 2; sc.view_settings.view_transform = 'AgX'
    sc["design"] = design; sc["q1_closed_deg"] = math.degrees(q_cl); sc["runner_travel_mm"] = S_MAX * 1000
    sc["move_time_s"] = T_mv; sc["timeline"] = str(dict(close=t0, closed=t1, open_start=t2, open=t3, dur=DUR))
    print("KIN", design, f"q1_closed {math.degrees(q_cl):.1f} travel {S_MAX*1000:.2f}mm move {T_mv:.2f}s dur {DUR:.2f}s",
          f"joints at closed {np.round(np.degrees(rib_pose(q_cl)[0]), 1)} fabric {GSM*area:.1f} g")

    bpy.ops.wm.save_as_mainfile(filepath=path)
    if os.environ.get("NOBAKE"): return
    if os.environ.get("BAKE_END"): cl.point_cache.frame_end = int(os.environ["BAKE_END"])
    with bpy.context.temp_override(scene=sc, point_cache=cl.point_cache):
        bpy.ops.ptcache.bake(bake=True)
    bpy.ops.wm.save_as_mainfile(filepath=path)

# ---------------- render ----------------
def render(path, cam, outdir, pct, frames):
    bpy.ops.wm.open_mainfile(filepath=path); sc = bpy.context.scene
    sc.camera = bpy.data.objects[cam]; sc.render.resolution_percentage = int(pct)
    sc.eevee.taa_render_samples = int(os.environ.get("SAMPLES", 10))
    for o in bpy.data.objects:
        if o.name.startswith("Touch_"): o.hide_render = o.name != f"Touch_{cam}"
    fl = range(*(lambda a, b: (a, b + 1))(*map(int, frames.split("-")))) if "-" in frames else [int(x) for x in frames.split(",")]
    for fr in fl:
        sc.frame_set(fr); sc.render.filepath = f"{outdir}/{cam}_{fr:04d}.png"
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    build(a[1], a[2]) if a[0] == "build" else render(*a[1:6])
