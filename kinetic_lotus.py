"""Kinetic lotus garment. Blender 5.2 LTS (bpy).
build:   python kinetic_lotus.py build OUT.blend
render:  python kinetic_lotus.py render IN.blend CAM OUTDIR PCT FRAMES   (FRAMES: "25-624" or "97,283")
Inside Blender:  blender -b -P kinetic_lotus.py -- <same args>
"""
import bpy, sys, math, os
from mathutils import Vector, Matrix

# ---------------- parameters (metres, seconds, degrees) ----------------
FPS, PRE, DUR = 24, 24, 25.0            # pre-roll frames for cloth settling, visible length
WAIST_Z, WAIST_W = 1.00, 0.34           # torso waist height and width
HUB_X = 0.05                            # hub offset to wearer's left (+X, wearer faces -Y)
RING_R, RING_IN, RING_T = 0.045, 0.037, 0.010
RING_Y = -0.165                         # ring centre plane
BOX = (0.05, 0.04, 0.05)                # motor box x, depth(y), z
PLATE_R, PLATE_T, PLATE_Y0 = 0.035, 0.003, -0.140
N_RIB, N_BONE = 8, 7
RIB_L, RIB_W, RIB_T = 0.21, 0.008, 0.0015
LINE_R, LINE_OFS = 0.0003, 0.002        # line radius, line offset in front of rib
FAB_R, FAB_IN, FAB_OFS = 0.225, 0.060, 0.004   # fabric outer radius, centre opening, offset in front of rib
LOBES, LOBE_DEPTH, RIPPLES, RIPPLE_AMP = 8, 0.020, 24, 0.004
FAB_NT, FAB_NR = 160, 28                # fabric angular / radial subdivisions
PIN_HALF = 0.0045                       # fabric pinned within this distance of a rib centreline
CURL_OPEN, CURL_CLOSED = 8.0, 85.0      # tip curl, degrees
TILT = 2.2                              # wobble plate tilt a
SPIN_T = 6.0                            # seconds per wobble turn
H_MAX = 0.025                           # plate pull back
GAIN = (CURL_CLOSED - CURL_OPEN) / H_MAX  # deg per metre of line pull
T_TOUCH, T_CLOSE, T_HOLD, T_OPEN = 10.0, 1.5, 2.0, 2.5

C = Vector((HUB_X, RING_Y, WAIST_Z))
Y_RIB = RING_Y - RING_T / 2 - RIB_T / 2
TH = [2 * math.pi * k / N_RIB for k in range(N_RIB)]
LB = RIB_L / N_BONE
W_BONE = [(i + 1) / sum(range(1, N_BONE + 1)) for i in range(N_BONE)]   # bend grows toward tip
FRONT = Vector((0, -1, 0))

def radial(th): return Vector((math.cos(th), 0, math.sin(th)))
def tang(th): return Vector((-math.sin(th), 0, math.cos(th)))
def ease(x): x = min(max(x, 0), 1); return 0.5 - 0.5 * math.cos(math.pi * x)

def h_of(t):
    t0, t1, t2, t3 = T_TOUCH, T_TOUCH + T_CLOSE, T_TOUCH + T_CLOSE + T_HOLD, T_TOUCH + T_CLOSE + T_HOLD + T_OPEN
    if t < t0 or t > t3: return 0.0
    if t < t1: return H_MAX * ease((t - t0) / T_CLOSE)
    if t < t2: return H_MAX
    return H_MAX * (1 - ease((t - t2) / T_OPEN))

# ---------------- helpers ----------------
def mat(name, col, rough=0.6, metal=0.0, sheen=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Sheen Weight"].default_value = sheen
    return m

def obj(name, verts, faces, m, smooth=False, mw=Matrix()):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in verts], [], faces); me.update()
    if smooth: me.shade_smooth()
    o = bpy.data.objects.new(name, me); o.data.materials.append(m); o.matrix_world = mw
    bpy.context.collection.objects.link(o); return o

def ring_mesh(rin, rout, th, n=64):  # annulus / disc (rin=0) along local Z
    v, f = [], []
    for z in (-th / 2, th / 2):
        for r in (rin, rout):
            v += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    q = lambda a, b, i: (a * n + i, a * n + (i + 1) % n, b * n + (i + 1) % n, b * n + i)
    for i in range(n): f += [q(0, 1, i), q(1, 3, i), q(3, 2, i), q(2, 0, i)]
    return v, f

def box_mesh(sx, sy, sz):
    v = [(sx * (i & 1 and .5 or -.5), sy * (i & 2 and .5 or -.5), sz * (i & 4 and .5 or -.5)) for i in range(8)]
    return v, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]

def torus_mesh(R, r, nu=16, nv=6):
    v = [((R + r * math.cos(2 * math.pi * j / nv)) * math.cos(2 * math.pi * i / nu),
          (R + r * math.cos(2 * math.pi * j / nv)) * math.sin(2 * math.pi * i / nu),
          r * math.sin(2 * math.pi * j / nv)) for i in range(nu) for j in range(nv)]
    f = [(i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv)
         for i in range(nu) for j in range(nv)]
    return v, f

def frame(origin, z_axis, x_axis):  # matrix with local Z along z_axis
    z = z_axis.normalized(); x = (x_axis - x_axis.dot(z) * z).normalized(); y = z.cross(x)
    m = Matrix((x, y, z)).transposed().to_4x4(); m.translation = origin; return m

def bone_weights(s):  # linear blend between neighbouring bones, smooth bend
    u = min(max(s / LB - 0.5, 0), N_BONE - 1); i0 = int(u); fr = u - i0
    return [(i0, 1 - fr)] + ([(i0 + 1, fr)] if i0 + 1 < N_BONE else [])

def driver(idb, path, idx, expr, vars):
    d = idb.driver_add(path, idx).driver; d.type = 'SCRIPTED'
    for n, (tid, p) in vars.items():
        v = d.variables.new(); v.name = n; v.targets[0].id = tid; v.targets[0].data_path = p
    d.expression = expr; assert d.is_simple_expression, expr   # no Python needed, safe on file open

# ---------------- build ----------------
def build(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    sc = bpy.context.scene
    sc.render.fps, sc.frame_start, sc.frame_end = FPS, PRE + 1, PRE + int(DUR * FPS)
    M_DARK = mat("mech_dark_grey", (0.045, 0.047, 0.05), 0.55)
    M_FAB = mat("fabric_off_white", (0.80, 0.78, 0.73), 0.85, sheen=0.4)
    M_LINE = mat("fishing_line", (0.75, 0.77, 0.78), 0.2)
    M_TORSO = mat("torso", (0.13, 0.125, 0.12), 0.7)
    M_MARK = mat("index_mark", (0.6, 0.6, 0.6), 0.4)
    M_TEXT = mat("touch_text", (1, 1, 1), 1.0)
    M_TEXT.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 1, 1, 1)
    M_TEXT.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.5

    # torso: lofted super ellipses (z, half width, half depth)
    secs = [(0.70, 0.165, 0.11), (0.80, 0.19, 0.12), (0.90, 0.18, 0.112), (WAIST_Z, WAIST_W / 2, 0.105),
            (1.10, 0.168, 0.108), (1.20, 0.175, 0.12), (1.28, 0.185, 0.135), (1.36, 0.195, 0.12),
            (1.43, 0.20, 0.10), (1.475, 0.15, 0.08), (1.50, 0.065, 0.055), (1.58, 0.058, 0.05)]
    n, e = 64, 2.6; v, f = [], []
    for z, a, b in secs:
        for i in range(n):
            c, s = math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)
            v.append((a * math.copysign(abs(c) ** (2 / e), c), b * math.copysign(abs(s) ** (2 / e), s), z))
    for j in range(len(secs) - 1):
        f += [(j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i) for i in range(n)]
    v += [(0, 0, secs[0][0]), (0, 0, secs[-1][0])]
    f += [(len(v) - 2, (i + 1) % n, i) for i in range(n)] + [(len(v) - 1, (len(secs) - 1) * n + i, (len(secs) - 1) * n + (i + 1) % n) for i in range(n)]
    torso = obj("Torso", v, f, M_TORSO, True)
    torso.modifiers.new("Subsurf", 'SUBSURF').levels = 2
    torso.modifiers.new("Collision", 'COLLISION'); torso.collision.thickness_outer = 0.003

    # hub: ring, motor box, spokes, standoffs to torso
    ax = frame(C, Vector((0, 1, 0)), Vector((1, 0, 0)))
    obj("HubRing", *ring_mesh(RING_IN, RING_R, RING_T), M_DARK, mw=ax)
    obj("MotorBox", *box_mesh(*BOX), M_DARK, mw=Matrix.Translation(C))
    for j in range(4):
        th = math.radians(22.5 + 90 * j); mid = C + radial(th) * (RING_IN + 0.025) / 2
        obj(f"Spoke{j}", *box_mesh(0.012, 0.004, 0.004), M_DARK, mw=frame(mid, FRONT, radial(th)) @ Matrix.Rotation(math.pi / 2, 4, 'Z'))
        p = C + radial(th) * 0.041 + Vector((0, RING_T / 2, 0)); ln = -0.10 - p.y + 0.01
        obj(f"Standoff{j}", *ring_mesh(0, 0.003, ln, 16), M_DARK, mw=frame(p + Vector((0, ln / 2, 0)), Vector((0, 1, 0)), Vector((1, 0, 0))))
    for k, th in enumerate(TH):  # hinge cheeks and pivot pin at each rib base
        base = C + radial(th) * RING_R + Vector((0, Y_RIB - C.y, 0))
        for sgn in (-1, 1):
            obj(f"Clevis{k}{sgn}", *box_mesh(0.002, 0.007, 0.008), M_DARK,
                mw=frame(base + tang(th) * sgn * 0.0055, radial(th), FRONT))
        obj(f"Pin{k}", *ring_mesh(0, 0.0012, 0.013, 12), M_DARK, mw=frame(base, tang(th), FRONT))
        e_ = C + radial(th) * 0.041 + Vector((0, -RING_T / 2 - 0.001, 0))
        obj(f"HubEyelet{k}", *torus_mesh(0.0016, 0.0005), M_DARK, mw=frame(e_, Vector((0, 1, 0)), radial(th)))

    # wobble plate: Base (pull h) > Wob (spin -phi) > Tilt (a) > Disc (spins) / Collar (unspun, holds lines)
    ctrl = bpy.data.objects.new("Mechanism", None); sc.collection.objects.link(ctrl)
    for k_, val in dict(h=0.0, phi=0.0, tilt=0.0, gain=GAIN, curl_open=0.0).items():   # rest pose while rigging
        ctrl[k_] = val
    emp = {}
    for nm, par in (("SwashBase", None), ("SwashWob", "SwashBase"), ("SwashTilt", "SwashWob"), ("SwashCollar", "SwashTilt")):
        e_ = bpy.data.objects.new(nm, None); sc.collection.objects.link(e_); e_.empty_display_size = 0.02
        if par: e_.parent = emp[par]
        emp[nm] = e_
    emp["SwashBase"].location = (C.x, PLATE_Y0, C.z)
    V = lambda p: {"h": (ctrl, '["h"]'), "p": (ctrl, '["phi"]'), "a": (ctrl, '["tilt"]'), "g": (ctrl, '["gain"]'), "c0": (ctrl, '["curl_open"]')}
    driver(emp["SwashBase"], "location", 1, f"{PLATE_Y0}+h", V(0))
    driver(emp["SwashWob"], "rotation_euler", 1, "-p", V(0))
    driver(emp["SwashTilt"], "rotation_euler", 2, "a", V(0))
    driver(emp["SwashCollar"], "rotation_euler", 1, "p", V(0))
    yax = frame(Vector(), Vector((0, 1, 0)), Vector((1, 0, 0)))
    for nm, (vv, ff), par in (("WobblePlate", ring_mesh(0.004, PLATE_R - 0.003, PLATE_T), "SwashTilt"),
                              ("PlateCollar", ring_mesh(PLATE_R - 0.003, PLATE_R + 0.001, PLATE_T + 0.001), "SwashCollar"),
                              ("PlateMark", box_mesh(0.016, PLATE_T + 0.002, 0.004), "SwashTilt"),
                              ("Shaft", ring_mesh(0, 0.004, 0.03), "SwashCollar")):
        o = obj(nm, vv, ff, M_MARK if nm == "PlateMark" else M_DARK); o.parent = emp[par]
        o.matrix_basis = {"PlateMark": Matrix.Translation((0.018, 0, 0)), "Shaft": Matrix.Translation((0, -0.015, 0)) @ yax}.get(nm, yax)
    for k, th in enumerate(TH):
        o = obj(f"PlateEyelet{k}", *torus_mesh(0.0015, 0.0005), M_DARK); o.parent = emp["SwashCollar"]
        o.matrix_basis = frame(radial(th) * PLATE_R, tang(th), FRONT)

    # armature: 8 chains of N_BONE bones, Z axis faces front so +X rotation curls forward
    arm = bpy.data.objects.new("RibRig", bpy.data.armatures.new("RibRig")); sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='EDIT')
    for k, th in enumerate(TH):
        prev = None
        for i in range(N_BONE):
            b = arm.data.edit_bones.new(f"R{k}_{i}")
            b.head = C + Vector((0, Y_RIB - C.y, 0)) + radial(th) * (RING_R + i * LB)
            b.tail = b.head + radial(th) * LB; b.align_roll(FRONT)
            if prev: b.parent, b.use_connect = prev, True
            prev = b
    bpy.ops.object.mode_set(mode='OBJECT')
    for k, th in enumerate(TH):
        for i in range(N_BONE):
            pb = arm.pose.bones[f"R{k}_{i}"]; pb.rotation_mode = 'XYZ'
            driver(arm, f'pose.bones["R{k}_{i}"].rotation_euler', 0,
                   f"{W_BONE[i]:.6f}*radians(c0+g*(h+{PLATE_R}*tan(a)*cos({th:.6f}-p)))", V(0))

    def rig(o):
        for k in range(N_RIB):
            for i in range(N_BONE): o.vertex_groups.new(name=f"R{k}_{i}")
        o.modifiers.new("Armature", 'ARMATURE').object = arm

    # ribs: flat strips, subdivided along length, smoothly skinned
    v, f, wts, ns = [], [], [], 42
    for k, th in enumerate(TH):
        o0 = len(v)
        for j in range(ns + 1):
            s = RIB_L * j / ns
            for w, d in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                v.append(C + Vector((0, Y_RIB - C.y, 0)) + radial(th) * (RING_R + s) + tang(th) * w * RIB_W / 2 + FRONT * d * RIB_T / 2)
                wts.append((k, s))
        for j in range(ns):
            a_, b_ = o0 + 4 * j, o0 + 4 * (j + 1)
            f += [(a_ + q, a_ + (q + 1) % 4, b_ + (q + 1) % 4, b_ + q) for q in range(4)]
        f += [(o0 + 3, o0 + 2, o0 + 1, o0), (o0 + 4 * ns, o0 + 4 * ns + 1, o0 + 4 * ns + 2, o0 + 4 * ns + 3)]
    ribs = obj("Ribs", v, f, M_DARK); rig(ribs)
    for vi, (k, s) in enumerate(wts):
        for i, w in bone_weights(s): ribs.vertex_groups[f"R{k}_{i}"].add([vi], w, 'REPLACE')
    ribs.modifiers.new("Collision", 'COLLISION'); ribs.collision.thickness_outer = 0.001

    bpy.context.view_layer.update()
    # tip eyelets ride on the last bone
    for k, th in enumerate(TH):
        o = obj(f"TipEyelet{k}", *torus_mesh(0.0014, 0.0004), M_DARK)
        o.parent, o.parent_type, o.parent_bone = arm, 'BONE', f"R{k}_{N_BONE - 1}"
        bpy.context.view_layer.update()
        o.matrix_world = frame(C + Vector((0, Y_RIB - C.y, 0)) + radial(th) * (RING_R + RIB_L - 0.01) + FRONT * LINE_OFS, radial(th), FRONT)

    # fishing lines: poly curves, every point hooked to the part it rides on
    collar = emp["SwashCollar"]; bpy.context.view_layer.update()
    for k, th in enumerate(TH):
        on_rib = lambda s: C + Vector((0, Y_RIB - C.y, 0)) + radial(th) * (RING_R + s) + FRONT * LINE_OFS
        pts = [(on_rib(RIB_L - 0.01), f"R{k}_{N_BONE - 1}")] + [(on_rib((i + 0.5) * LB), f"R{k}_{i}") for i in range(N_BONE - 2, -1, -1)]
        pts += [(C + radial(th) * 0.041 + Vector((0, -RING_T / 2 - 0.001, 0)), None),
                (C + radial(th) * 0.041 + Vector((0, RING_T / 2 + 0.001, 0)), None),
                (collar.matrix_world @ (radial(th) * PLATE_R), "COLLAR")]
        cu = bpy.data.curves.new(f"Line{k}", 'CURVE'); cu.dimensions = '3D'
        cu.bevel_depth, cu.bevel_resolution = LINE_R, 2
        sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
        for p, (co, _) in zip(sp.points, pts): p.co = (*co, 1)
        lo = bpy.data.objects.new(f"Line{k}", cu); lo.data.materials.append(M_LINE); sc.collection.objects.link(lo)
        for idx, (co, tgt) in enumerate(pts):
            if tgt is None: continue
            hm = lo.modifiers.new(f"hook{idx}", 'HOOK'); hm.falloff_type = 'NONE'
            if tgt == "COLLAR":
                hm.object = collar; hm.matrix_inverse = collar.matrix_world.inverted()
            else:
                hm.object, hm.subtarget = arm, tgt
                hm.matrix_inverse = (arm.matrix_world @ arm.data.bones[tgt].matrix_local).inverted()
            hm.vertex_indices_set([idx])

    # fabric: polar grid annulus with scalloped, rippled edge
    v, f, wts, pin = [], [], [], []
    for j in range(FAB_NT):
        th = 2 * math.pi * j / FAB_NT
        Rt = FAB_R - LOBE_DEPTH * (1 - math.cos(LOBES * th)) / 2
        for i in range(FAB_NR + 1):
            u = i / FAB_NR; r = FAB_IN + (Rt - FAB_IN) * u
            p = C + Vector((0, Y_RIB - C.y, 0)) + radial(th) * r + FRONT * (FAB_OFS + RIPPLE_AMP * math.sin(RIPPLES * th) * u ** 2)
            v.append(p)
            k = round(th / (2 * math.pi / N_RIB)) % N_RIB
            dist = abs(r * math.sin(th - TH[k]))
            pin.append(dist < PIN_HALF and r - RING_R <= RIB_L); wts.append((k, r - RING_R))
    for j in range(FAB_NT):
        a_, b_ = j * (FAB_NR + 1), ((j + 1) % FAB_NT) * (FAB_NR + 1)
        f += [(a_ + i, b_ + i, b_ + i + 1, a_ + i + 1) for i in range(FAB_NR)]
    fab = obj("Fabric", v, f, M_FAB, True); rig(fab)
    gp, gf = fab.vertex_groups.new(name="pin"), fab.vertex_groups.new(name="free")
    for vi, ((k, s), pn) in enumerate(zip(wts, pin)):
        (gp if pn else gf).add([vi], 1.0, 'REPLACE')
        if pn:
            for i, w in bone_weights(s): fab.vertex_groups[f"R{k}_{i}"].add([vi], w, 'REPLACE')
    cl = fab.modifiers.new("Cloth", 'CLOTH'); cs = cl.settings
    cs.quality, cs.mass, cs.air_damping = 12, 0.004, 2.0
    cs.bending_model = 'ANGULAR'
    cs.tension_stiffness = cs.compression_stiffness = 60.0; cs.shear_stiffness = 30.0; cs.bending_stiffness = 6.0
    cs.tension_damping = cs.compression_damping = cs.shear_damping = 5.0; cs.bending_damping = 1.0
    cs.vertex_group_mass, cs.pin_stiffness = "pin", 1.0
    cc = cl.collision_settings
    cc.collision_quality, cc.distance_min, cc.use_self_collision, cc.self_distance_min = 4, 0.002, True, 0.0015
    cc.vertex_group_object_collisions = "free"
    cl.point_cache.frame_start, cl.point_cache.frame_end = 1, sc.frame_end
    fab.modifiers.new("Subsurf", 'SUBSURF').levels = 1
    fab.modifiers.new("Solidify", 'SOLIDIFY').thickness = 0.0006

    # motion: only the plate pose is animated; ribs, lines, fabric follow through drivers / hooks / cloth
    for fr in range(1, sc.frame_end + 1):
        t = (fr - PRE - 1) / FPS
        ctrl["h"], ctrl["phi"] = h_of(t), 2 * math.pi * t / SPIN_T
        ctrl.keyframe_insert('["h"]', frame=fr); ctrl.keyframe_insert('["phi"]', frame=fr)
    for fr, ramp in ((1, 0.0), (PRE // 2, 1.0)):   # pre-roll: ease mechanism from flat rest into the open pose
        ctrl["curl_open"], ctrl["tilt"] = CURL_OPEN * ramp, math.radians(TILT) * ramp
        ctrl.keyframe_insert('["curl_open"]', frame=fr); ctrl.keyframe_insert('["tilt"]', frame=fr)

    # studio: world, lights, cameras with "touch" overlay
    w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.22, 0.22, 0.23, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    for nm, loc, size, en in (("Key", (1.6, -1.8, 2.2), 1.5, 110), ("Fill", (-1.8, -1.4, 1.3), 2.0, 35),
                              ("Rim", (0.6, 1.6, 2.0), 1.0, 70), ("Low", (0.4, -1.0, 0.2), 1.2, 12)):
        L = bpy.data.lights.new(nm, 'AREA'); L.size, L.energy = size, en
        lo = bpy.data.objects.new(nm, L); sc.collection.objects.link(lo); lo.location = loc
        lo.rotation_euler = (Vector((C.x, -0.2, 1.05)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    for nm, loc, tgt, lens in (("CamFront3Q", (1.15, -1.20, 1.25), (0.06, -0.26, 1.03), 70),
                               ("CamSide", (1.55, -0.24, 1.08), (0.05, -0.21, 1.0), 60),
                               ("CamHub", (0.21, -0.128, 0.93), (0.05, -0.14, 1.0), 35)):
        cam = bpy.data.objects.new(nm, bpy.data.cameras.new(nm)); sc.collection.objects.link(cam)
        cam.data.lens, cam.data.clip_start = lens, 0.005; cam.location = loc
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        d = 0.05; hh = d * 18 / lens
        tx = bpy.data.objects.new(f"Touch_{nm}", bpy.data.curves.new(f"Touch_{nm}", 'FONT')); sc.collection.objects.link(tx)
        tx.data.body, tx.data.size, tx.data.align_x = "touch", 0.09 * hh, 'CENTER'
        tx.parent = cam; tx.location = (0, -0.82 * hh, -d)
        tx.data.materials.append(M_TEXT)
        for t, val in ((T_TOUCH - 0.15, 0), (T_TOUCH, 1), (T_TOUCH + T_CLOSE + T_HOLD + T_OPEN, 1), (T_TOUCH + T_CLOSE + T_HOLD + T_OPEN + 0.15, 0)):
            tx.scale = (val,) * 3; tx.keyframe_insert("scale", frame=PRE + 1 + t * FPS)
    sc.camera = bpy.data.objects["CamFront3Q"]
    r = sc.render; r.engine = 'BLENDER_EEVEE'; r.resolution_x, r.resolution_y = 1080, 1350
    sc.eevee.taa_render_samples = 10; sc.eevee.shadow_ray_count = 2; sc.view_settings.view_transform = 'AgX'

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
    if "-" in frames: a, b = map(int, frames.split("-")); fl = range(a, b + 1)
    else: fl = [int(x) for x in frames.split(",")]
    for fr in fl:
        sc.frame_set(fr); sc.render.filepath = f"{outdir}/{cam}_{fr:04d}.png"
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    build(a[1]) if a[0] == "build" else render(*a[1:6])
