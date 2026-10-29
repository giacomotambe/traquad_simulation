"""Replay of a real traquad run in Isaac Sim, through the replica of the real controllers.

The joystick twist of a recorded run (traquad_bags, converted with real_id/bag2npz.py) goes through
traquad_real_controller.py exactly as on the robot: R2 posture (when it was on), omni_controller (2 Hz low-pass,
TraquadIK from the simulated shoulder angles, wheel compensation) and the moteus drives:
- HFE: PD Kp 60 / Kd 0.6 on the posture references (PhysX drive, limit --hip_max);
- tracks: velocity loop Kd 0.9 (Kp 0) as the PhysX velocity drive of the wheels (implicit, as the fast loop of the
  moteus; a loop computed here, --drive explicit, makes the contacts slip), command clipped at 25 rad/s, plus the
  effort of the wheel compensation; the 45 mm sprocket (WHEEL_RAD) is referred to the 15 mm wheels;
- internal friction of the track (belt, sprocket, gear), at the track joint: Coulomb --track_friction [N m] and
  viscous --track_viscous [N m s/rad], PhysX joint friction of the wheels (--friction_model effort: as wheel effort);
- the HFE offset of the asset (finalize_usd.py --hfe_offset) makes the simulated hip angles the robot's encoder angles;
- sprocket torque on the ankle: off (factor --grid sprocket=..., see README).
The ankle law in the air is the one of ankle_control.py (PI towards the sprocket speed).

Several robots can run side by side with different parameters (--grid "name=v1,v2;name2=..."; cartesian product),
all with the same commands. Logs per robot at 50 Hz (npz): base pose and body velocities, track joint speeds and
torques, HFE and ankle angles, controller commands.

usage: ./isaac.sh real_replay.py --data <run.npz> --preset sim_ident --t0 745 --t1 860 --r2 747.1-900 --out out.npz
       [--grid "roller_friction=0.06,0.075;hfe_offset=0,-0.05"] [--wheel_source direct --hfe_source log]
"""
import argparse
import itertools
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='run converted by real_id/bag2npz.py')
parser.add_argument('--out', required=True)
parser.add_argument('--usd', default=None, help='robot USD (default: assets/traquad/traquad.usda)')
parser.add_argument('--preset', default='current', help='controller settings of the run (traquad_real_controller.config)')
parser.add_argument('--t0', type=float, default=0.0, help='start of the replay [s of the robot bag]')
parser.add_argument('--t1', type=float, default=None, help='end of the replay')
parser.add_argument('--r2', default='', help='R2 intervals "a-b,c-d" [s of the robot bag] (posture on)')
parser.add_argument('--wheel_source', default='ik', choices=['ik', 'direct'],
                    help='track commands: from the twist through the IK, or the direct track commands logged on the robot '
                         '(identification missions, wheel_mode direct)')
parser.add_argument('--hfe_source', default='replica', choices=['replica', 'log'],
                    help='hip references: replica of the joystick (R2) or the ones logged on the robot')
parser.add_argument('--dt', type=float, default=0.002, help='physics and control step (the SBC runs at 500 Hz)')
parser.add_argument('--settle', type=float, default=2.0, help='seconds standing before the replay')
parser.add_argument('--mu', type=float, default=0.45,
                    help='wheel-ground friction (static = dynamic unless --mu_dyn); 0.45 identified on the lab mat')
parser.add_argument('--mu_dyn', type=float, default=None)
parser.add_argument('--roller_friction', type=float, default=0.10)
parser.add_argument('--roller_damping', type=float, default=1e-4)
parser.add_argument('--track_friction', type=float, default=1.16, help='Coulomb friction of the track joint [N m]')
parser.add_argument('--track_viscous', type=float, default=0.013, help='viscous friction of the track joint [N m s/rad]')
parser.add_argument('--track_armature', type=float, default=None, help='armature per wheel (default: USD)')
parser.add_argument('--track_max', type=float, default=10.0 * (40.0 * 15.0) / (80.0 * 9.0),
                    help='track joint torque limit [N m] (0.83 x the hip limit: same actuator)')
parser.add_argument('--hip_max', type=float, default=10.0)
parser.add_argument('--ankle_kd', type=float, default=None)
parser.add_argument('--hip_kp', type=float, default=60.0, help='HFE PD stiffness [N m/rad]')
parser.add_argument('--hfe_offset', type=float, default=0.0,
                    help='geometric leg angle - encoder angle [rad] (alpha_sim = alpha_encoder + offset), on top of the '
                         'offset written in the USD (finalize_usd.py --hfe_offset, -0.05)')
parser.add_argument('--sprocket_fwd', type=float, default=0.0,
                    help='extra sprocket torque factor while the track drives forward (belt tension on one run)')
parser.add_argument('--sprocket_bwd', type=float, default=0.0, help='extra sprocket torque factor while it drives backward')
parser.add_argument('--ankle_friction', type=float, default=0.0,
                    help='Coulomb friction of the ankle pivot [N m]: turning the frame about the sprocket moves the belt and '
                         'backdrives the track gearbox, so on the robot the track stays where it is with the motor off')
parser.add_argument('--slip_bwd', type=float, default=0.0,
                    help='extra slip of a track driven backward while turning: its ground speed is scaled by '
                         '1 - slip_bwd * load (load 0 straight, 1 in place, as the IK): measured on the robot (the track '
                         'driving backward slips about twice as much as the forward one), source of the forward creep')
parser.add_argument('--frames', default=None,
                    help='also log the full pose (base and every joint) 30 times per simulated second into this npz, for '
                         'render_replay.py (one robot only)')
parser.add_argument('--slope', type=float, default=0.0,
                    help='climb test: gravity tilts to this slope [deg] over 2 s from the start of the replay, +x uphill '
                         '(the robot starts facing +x)')
parser.add_argument('--com_x', type=float, default=0.0, help='shift of the base_link centre of mass along x [m]')
parser.add_argument('--r2_sign', type=float, default=1.0, help='test: sign of the R2 posture offset in the simulation')
parser.add_argument('--contact_stiffness', type=float, default=0.0,
                    help='compliant contact stiffness of the robot material [N/m] (rubber belt; 0 = rigid contact)')
parser.add_argument('--contact_damping', type=float, default=0.0, help='compliant contact damping [N s/m]')
parser.add_argument('--friction_model', default='physx', choices=['effort', 'physx'],
                    help='internal track friction as explicit wheel effort or as PhysX joint friction of the wheels')
parser.add_argument('--variants', default='',
                    help='one robot per variant instead of the grid: "-|mu=0.6|track_friction=0,track_viscous=0"')
parser.add_argument('--grid', default='', help='per-robot parameters, e.g. "roller_friction=0.02,0.06;mu_dyn=0.5,0.6"')
parser.add_argument('--spacing', type=float, default=30.0)
parser.add_argument('--material', default='per_robot', choices=['per_robot', 'simple'],
                    help='simple: one material (--mu) on ground and robots, as track_test.py (no per-robot mu)')
parser.add_argument('--pos_iters', type=int, default=None, help='articulation solver position iterations')
parser.add_argument('--vel_iters', type=int, default=None, help='articulation solver velocity iterations')
parser.add_argument('--sprocket', type=float, default=0.0,
                    help='sprocket torque on the ankle on the ground, x this factor (ankle_control.py model; 0 = off). '
                         'With r_s = 45 mm it tips the tracks onto their stops when the robot accelerates (2026-10-26)')
parser.add_argument('--drive', default='implicit', choices=['explicit', 'implicit'],
                    help='track Kd loop computed here (explicit) or as PhysX velocity drive of the wheels (implicit)')
args, _ = parser.parse_known_args()

GRID_KEYS = ('mu', 'mu_dyn', 'roller_friction', 'roller_damping', 'track_friction', 'track_viscous', 'track_armature',
             'ankle_kd', 'contact_stiffness', 'contact_damping', 'sprocket', 'hip_kp', 'hfe_offset', 'com_x',
             'sprocket_fwd', 'sprocket_bwd', 'slip_bwd', 'ankle_friction')
grid = {}
for item in filter(None, args.grid.split(';')):
    k, v = item.split('=')
    assert k in GRID_KEYS, k
    grid[k] = [float(x) for x in v.split(',')]
combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())] if grid else [{}]
if args.variants:   # one robot per variant: the defaults with the overrides of that variant ('-' = defaults)
    combos = [{k: float(v) for k, v in (kv.split('=') for kv in item.split(',') if kv != '-')}
              for item in args.variants.split('|')]
    assert all(k in GRID_KEYS for c in combos for k in c), combos
PARAMS = [{k: c.get(k, getattr(args, k)) for k in GRID_KEYS} for c in combos]
for p in PARAMS:
    p['mu_dyn'] = p['mu'] if p['mu_dyn'] is None else p['mu_dyn']
NR = len(PARAMS)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import os  # noqa: E402
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import ankle_control as AC  # noqa: E402
import traquad_real_controller as rc  # noqa: E402

SimulationManager.switch_physics_engine('physx')
# no write-back of the simulation state to USD every step (the replay reads the tensor API only)
_settings = carb.settings.get_settings()
for _k in ('/physics/updateToUsd', '/physics/updateVelocitiesToUsd', '/physics/updateParticlesToUsd',
           '/physics/updateResidualsToUsd'):
    _settings.set(_k, False)
HERE = os.path.dirname(os.path.abspath(__file__))
USD = args.usd or os.path.join(HERE, 'assets/traquad/traquad.usda')
R = AC.WHEEL_RADIUS
RS = rc.WHEEL_RAD
TRACKS = ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')   # = LF, LH, RF, RH
HFE_NAMES = [f'{l}_HFE' for l in rc.LEGS]

# ── run data ────────────────────────────────────────────────────────────────────────────────────────────────────
D = np.load(args.data)
T0_BAG = D['js_t'][0]
tw_t, tw = D['tw_t'] - T0_BAG, D['tw'][:, [0, 2]]
ref_t = D['ref_t'] - T0_BAG
ref_names = list(D['ref_names'])
ref_q = D['ref_position'][:, [ref_names.index(n) for n in HFE_NAMES]]
ref_dq = D['ref_velocity'][:, [ref_names.index(n) for n in HFE_NAMES]]
t1 = args.t1 if args.t1 is not None else tw_t[-1]
if args.wheel_source == 'direct':
    dw_t = D['dw_t'] - T0_BAG
    dw_names = list(D['dw_names'])
    dw_v = D['dw_velocity'][:, [dw_names.index(f'{l}_WHEEL') for l in rc.LEGS]]
r2_iv = [tuple(float(x) for x in s.split('-')) for s in filter(None, args.r2.split(','))]


def r2_on(t):
    return any(a <= t < b for a, b in r2_iv)


js_t = D['js_t'] - T0_BAG
js_names = list(D['js_names'])
i0 = np.searchsorted(js_t, args.t0)
q_start = D['js_position'][i0, [js_names.index(n) for n in HFE_NAMES]]


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, '/World/Ground')
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')
    plane.CreateAxisAttr().Set('Z')
    plane.CreateWidthAttr().Set(1000.0)
    plane.CreateLengthAttr().Set(1000.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    # one material per robot; friction combine mode min everywhere and the ground at the largest friction of all,
    # so that each robot gets its own
    mats = []
    for k, p in enumerate(PARAMS + [{'mu': max(q['mu'] for q in PARAMS), 'mu_dyn': max(q['mu_dyn'] for q in PARAMS)}]):
        mat = UsdShade.Material.Define(stage, f'/World/Materials/m{k}')
        m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        m.CreateStaticFrictionAttr().Set(p['mu'])
        m.CreateDynamicFrictionAttr().Set(p['mu_dyn'])
        m.CreateRestitutionAttr().Set(0.0)
        if args.material == 'per_robot':
            pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
            pm.CreateFrictionCombineModeAttr().Set('min')
            if p.get('contact_stiffness'):
                pm.CreateCompliantContactStiffnessAttr().Set(p['contact_stiffness'])
                pm.CreateCompliantContactDampingAttr().Set(p['contact_damping'] or 0.0)
        mats.append(mat)
    if args.material == 'simple':
        mats = [mats[0]] * (NR + 1)
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mats[-1], UsdShade.Tokens.weakerThanDescendants, 'physics')
    for k in range(NR):
        path = f'/World/robot_{k}'
        stage_utils.add_reference_to_stage(usd_path=USD, path=path)
        prim = stage.GetPrimAtPath(path)
        prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(args.spacing * k, 0.0, 0.30))
    await app_utils.update_app_async()
    for k in range(NR):
        for p in stage.Traverse():
            if str(p.GetPath()).startswith(f'/World/robot_{k}/') and p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mats[k], UsdShade.Tokens.strongerThanDescendants, 'physics')
    SimulationManager.setup_simulation(dt=args.dt, device='cpu')
    for ps in stage.Traverse():
        if ps.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(ps).CreateSolverTypeAttr().Set('TGS')
    await app_utils.update_app_async()
    roots = []
    for k in range(NR):
        rr = [str(p.GetPath()) for p in stage.Traverse()
              if str(p.GetPath()).startswith(f'/World/robot_{k}/') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(rr[0]))
        art.CreateEnabledSelfCollisionsAttr().Set(False)
        if k == 0:
            print('solver iterations (USD):', art.GetSolverPositionIterationCountAttr().Get(),
                  art.GetSolverVelocityIterationCountAttr().Get(), flush=True)
        if args.pos_iters is not None:
            art.CreateSolverPositionIterationCountAttr().Set(args.pos_iters)
        if args.vel_iters is not None:
            art.CreateSolverVelocityIterationCountAttr().Set(args.vel_iters)
        roots += rr
    return Articulation(roots)


robot = app.run_coroutine(build())
app_utils.play()
app.update()

names = robot.dof_names
links = robot.link_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
hfe = [idx[n] for n in HFE_NAMES]
ankles = [idx[f"body_{t.split('_')[0].lower()}_{t.split('_')[1]}_ankle"] for t in TRACKS]
wheels = [[idx[f'joint_wheel_{k}_{t}'] for k in (1, 2, 3, 4)] for t in TRACKS]
wheel_links = [[links.index(f'wheel_{k}_{t}') for k in (1, 2, 3, 4)] for t in TRACKS]
all_wheels = sum(wheels, [])
NW = len(wheels[0])
rollers = [i for n, i in idx.items() if '_roller_' in n]
P = {k: np.array([p[k] if p[k] is not None else np.nan for p in PARAMS], np.float32) for k in GRID_KEYS}
print(f'robots {NR}, DOFs {N}, rollers {len(rollers)}, dt {args.dt}; params:', flush=True)
for k, p in enumerate(PARAMS):
    print(f'  robot {k}: ' + ' '.join(f'{a}={b}' for a, b in p.items()), flush=True)

# drives: HFE PD (PhysX), wheels and ankles by effort, rollers passive with dry friction
kp = np.zeros((NR, N), np.float32); kd = np.zeros((NR, N), np.float32); fmax = np.full((NR, N), 1e3, np.float32)
kp[:, hfe] = P['hip_kp'][:, None]; kd[:, hfe] = rc.HFE_KD; fmax[:, hfe] = args.hip_max
kd[:, ankles] = np.where(np.isnan(P['ankle_kd']), AC.PASSIVE_DAMPING, P['ankle_kd'])[:, None]
kd[:, rollers] = P['roller_damping'][:, None]
if args.drive == 'implicit':
    kd[:, all_wheels] = rc.TRACK_KD * (R / RS) ** 2 / NW
    fmax[:, all_wheels] = args.track_max * (R / RS) / NW
robot.set_dof_gains(stiffnesses=kp, dampings=kd)
robot.set_dof_max_efforts(fmax)
robot.set_dof_friction_properties(static_frictions=np.repeat(P['ankle_friction'][:, None], 4, 1),
                                  dynamic_frictions=np.repeat(P['ankle_friction'][:, None], 4, 1), dof_indices=ankles)
robot.set_dof_friction_properties(static_frictions=np.repeat(P['roller_friction'][:, None], len(rollers), 1),
                                  dynamic_frictions=np.repeat(P['roller_friction'][:, None], len(rollers), 1),
                                  dof_indices=rollers)
# internal friction of the track (belt, sprocket, gear) at the track joint: explicit, subtracted from the motor torque
# in the drive loop (PhysX joint friction on the wheels holds the robot back without loading the contacts)
FRICTION_EPS = 0.5   # [rad/s], smoothing of the Coulomb term
if True:   # PhysX joint friction of the wheels: from the parameters ('physx') or none ('effort', applied as effort)
    on = 1.0 if args.friction_model == 'physx' else 0.0
    c_w = on * P['track_friction'] * (R / RS) / NW
    b_w = on * P['track_viscous'] * (R / RS) ** 2 / NW
    robot.set_dof_friction_properties(static_frictions=np.repeat(c_w[:, None], len(all_wheels), 1),
                                      dynamic_frictions=np.repeat(c_w[:, None], len(all_wheels), 1),
                                      viscous_frictions=np.repeat(b_w[:, None], len(all_wheels), 1),
                                      dof_indices=all_wheels)
if not np.isnan(P['track_armature']).all():
    robot.set_dof_armatures(np.repeat(P['track_armature'][:, None], len(all_wheels), 1), dof_indices=all_wheels)

# centre of mass of the base moved along x (front/hind load split)
if np.any(P['com_x'] != 0):
    bl = [links.index('base_link')]
    cpos, cori = robot.get_link_coms(link_indices=bl)
    cpos = cpos.numpy().copy()
    cpos[:, 0, 0] += P['com_x']
    robot.set_link_coms(positions=cpos, link_indices=bl)

# initial state: hips as on the robot at t0, tracks flat
q0 = np.zeros((NR, N), np.float32)
q0[:, hfe] = q_start + rc.SIGMA[None] * P['hfe_offset'][:, None]
for j, t in enumerate(TRACKS):
    alpha = rc.SIGMA[j] * q_start[j] + P['hfe_offset']
    q0[:, ankles[j]] = (math.pi / 2 - alpha) * (1 if t.endswith('_F') else -1)
robot.set_dof_positions(q0)
robot.set_dof_position_targets(q0)
robot.set_dof_velocity_targets(np.zeros((NR, N), np.float32))

ctrls = [rc.RealRobotController(args.preset, r2=False, dt=args.dt) for _ in range(NR)]
ank_int = np.zeros((NR, 4)); air_time = np.zeros((NR, 4)); ank_air_prev = np.zeros((NR, 4), bool)
w_track_prev = None
# inertia of the track wheels referred to the track joint (armature of the asset, 0.001 kg m2 per wheel)
J_TRACK = np.where(np.isnan(P['track_armature']), 0.001, P['track_armature'])[:, None] * NW * (RS / R) ** 2
eff_idx = np.array(ankles + all_wheels)


ground_scale = np.ones((NR, 4))   # wheel (ground) speed / belt speed of each track, from the previous step


def track_state(dq):
    """Track joint speeds [rad/s] of the real robot from the wheel speeds: belt = -wheel * R (see the axes); the
    wheels carry the ground speed of the belt, ground_scale times the belt speed (slip_bwd)."""
    return np.stack([-dq[:, w].mean(1) * R / RS for w in wheels], 1) / ground_scale


def update_ground_scale(w_cmd):
    """slip_bwd: a track driven backward while the robot turns transfers less speed to the ground."""
    global ground_scale
    kap = np.array([1.0, 1.0, -1.0, -1.0])
    belt = kap * w_cmd * RS
    vl, vr = belt[:, :2].mean(1), belt[:, 2:].mean(1)
    sm, df = np.abs(vl + vr), np.abs(vr - vl)
    load = np.where(sm + df > 1e-6, 1 - sm / np.maximum(sm + df, 1e-9), 0.0)
    ground_scale = np.where(belt < 0, 1 - P['slip_bwd'][:, None] * load[:, None], 1.0)


def step_control(t, k_tw, dq, q):
    """Controllers -> efforts of ankles and wheels, HFE targets."""
    w_track = track_state(dq)
    tau_track = np.zeros((NR, 4)); w_cmd = np.zeros((NR, 4)); eff_cmd = np.zeros((NR, 4))
    q_ref = np.zeros((NR, 4)); dq_ref = np.zeros((NR, 4))
    for r_, c in enumerate(ctrls):
        c.set_r2(r2_on(t))
        c.step(q[r_, hfe] - rc.SIGMA * P['hfe_offset'][r_], dq[r_, hfe], w_track[r_])
        w_cmd[r_], eff_cmd[r_] = c.ctrl.wheel_cmd, c.ctrl.wheel_effort
        q_ref[r_], dq_ref[r_] = c.q_ref, c.dq_ref
        if args.r2_sign != 1.0:   # test: posture offset delta mirrored
            stand_q = rc.SIGMA * c.posture.c.stand
            q_ref[r_] = stand_q + args.r2_sign * (c.q_ref - stand_q)
            dq_ref[r_] = args.r2_sign * c.dq_ref
    if args.wheel_source == 'direct':
        k_ = np.searchsorted(dw_t, t) - 1
        w_cmd[:] = dw_v[k_] if k_ >= 0 else 0.0
        eff_cmd[:] = 0.0
    if args.hfe_source == 'log':
        q_ref[:] = [np.interp(t, ref_t, ref_q[:, j]) for j in range(4)]
        dq_ref[:] = [np.interp(t, ref_t, ref_dq[:, j]) for j in range(4)]
    q_ref = q_ref + rc.SIGMA[None] * P['hfe_offset'][:, None]
    # moteus: Kd only, feedforward effort, torque derated above max_velocity (+1.5 rad/s)
    tau = rc.TRACK_KD * (np.clip(w_cmd, -rc.TRACK_MAX_VELOCITY, rc.TRACK_MAX_VELOCITY) - w_track) + \
        np.clip(eff_cmd, -rc.MAX_FEEDFORWARD, rc.MAX_FEEDFORWARD)
    derate = np.clip(1 - (np.abs(w_track) - rc.TRACK_MAX_VELOCITY) / 1.5, 0, 1)
    tau = np.where(np.sign(tau) == np.sign(w_track), tau * derate, tau)
    tau_track = np.clip(tau, -args.track_max, args.track_max)
    tau_loss = P['track_friction'][:, None] * np.tanh(w_track / FRICTION_EPS) + P['track_viscous'][:, None] * w_track
    if args.friction_model == 'physx':
        tau_loss = np.zeros_like(tau_loss)   # in the PhysX joints
    return w_track, w_cmd, eff_cmd, tau_track, tau_loss, q_ref, dq_ref


def ankle_torques(tau_track, w_cmd, dq):
    """Ankle law: on the ground passive (+ sprocket torque, factor 0 by default); in the air the frame turns with the
    sprocket, driven by the controller of its track: same speed command, Kd of the track as PhysX drive (set in the
    main loop) and the integral of the wheel compensation (ki 3 N m/rad, +-1 N m) as effort."""
    global ank_int, air_time
    tf = robot._physics_articulation_view.get_link_transforms().numpy().reshape(NR, len(links), 7)
    off = np.stack([tf[:, wl, 2].min(1) > R + AC.CONTACT_MARGIN for wl in wheel_links], 1)
    air_time = np.where(off, air_time + args.dt, 0.0)
    in_air = air_time > AC.AIR_DELAY
    # ankle axes along +y; belt forward speed v = side * w * RS; ankle turns with the sprocket: w_ankle = -w (axes)
    left = np.array([t.startswith('LEFT') for t in TRACKS])
    kappa = np.where(left, 1.0, -1.0)
    err = kappa * w_cmd - dq[:, ankles]          # sprocket speed v_belt / r_s
    wc = ctrls[0].cfg.wheel_comp
    ank_int = np.where(in_air, np.clip(ank_int + wc.ki * err * args.dt, -wc.i_max, wc.i_max), 0.0)
    tau_air = ank_int
    # wheel torques (joint axes: left -y, right +y) and the sprocket torque on the ankle (+y): the drive torque of the
    # track joint enters the track frame through the pivot (ankle_control: side * sum(wheel torques) * r_s / r)
    side = np.where(left, -1.0, 1.0)
    tw_sum = -tau_track * (R / RS)               # sum of the wheel torques of each track, wheel axes (belt fwd = -w)
    # the sprocket puts on the belt only the part of the motor torque that becomes belt force: minus the internal
    # friction (when it is PhysX joint friction it is not in tau_track yet) and minus the torque that accelerates the
    # track wheels (their armature stands for the rotor and belt inertia, which on the robot does not load the belt)
    global w_track_prev
    w_tr = track_state(dq)
    alpha = (w_tr - w_track_prev) / args.dt if w_track_prev is not None else np.zeros_like(w_tr)
    w_track_prev = w_tr
    belt = tau_track - J_TRACK * alpha
    if args.friction_model == 'physx':
        belt = belt - (P['track_friction'][:, None] * np.tanh(w_tr / FRICTION_EPS) + P['track_viscous'][:, None] * w_tr)
    sprocket = side * (-belt * (R / RS)) * RS / R
    # factor of the sprocket torque, by drive direction of each track (forward belt = kappa * w_cmd > 0)
    fwd = kappa * w_cmd > 0
    sprocket = sprocket * (P['sprocket'][:, None] + np.where(fwd, P['sprocket_fwd'][:, None], P['sprocket_bwd'][:, None]))
    return np.where(in_air, tau_air, sprocket), tw_sum / NW, in_air, kappa * w_cmd


def get_state():
    pos, quat = robot.get_world_poses()
    lin, ang = robot.get_velocities()
    return pos.numpy(), quat.numpy(), lin.numpy(), ang.numpy()


log = {k: [] for k in ('t', 'pos', 'yaw', 'vx', 'vy', 'wz', 'roll', 'pitch', 'w_track', 'w_cmd', 'eff_cmd', 'tau_track',
                       'q_hfe', 'q_ref', 'q_ankle', 'in_air', 'wheel_z', 'body_z', 'roller_w')}
track_links = [links.index(f"body_{t.split('_')[0].lower()}_{t.split('_')[1]}") for t in TRACKS]
roller_of = [[i for n, i in idx.items() if '_roller_' in n and n.split('_roller_')[0].endswith(t)] for t in TRACKS]
log_every = max(1, int(round(0.02 / args.dt)))

FPS = 30   # --frames
frames = {k: [] for k in ('t', 'pos', 'quat', 'q', 'vx', 'wz', 'cmd', 'r2')}
k_tw = np.searchsorted(tw_t, args.t0) - 1
n_settle = int(round(args.settle / args.dt))
n_run = int(round((t1 - args.t0) / args.dt))
wall = time.time()
for k in range(n_settle + n_run):
    t = args.t0 + (k - n_settle) * args.dt
    if args.slope and k >= n_settle:
        th_ = math.radians(args.slope) * min(1.0, (k - n_settle) * args.dt / 2.0)
        if k == n_settle or (k - n_settle) * args.dt <= 2.0 + args.dt:
            SimulationManager.get_physics_scenes()[0].set_gravity(Gf.Vec3f(-9.81 * math.sin(th_), 0.0,
                                                                           -9.81 * math.cos(th_)))
    if k >= n_settle:
        while k_tw + 1 < len(tw_t) and tw_t[k_tw + 1] <= t:
            k_tw += 1
            for c in ctrls:
                c.set_twist(*tw[k_tw])
    else:
        for c in ctrls:
            c.set_twist(0.0, 0.0)
    q = robot.get_dof_positions().numpy()
    dq = robot.get_dof_velocities().numpy()
    if np.isnan(dq).any():
        print('NaN at', t, flush=True)
        break
    w_track, w_cmd, eff_cmd, tau_track, tau_loss_now, q_ref, dq_ref = step_control(max(t, args.t0), k_tw, dq, q)
    update_ground_scale(w_cmd)
    w_cmd_now, eff_cmd_now = w_cmd, eff_cmd
    # the internal friction acts between belt and track frame: the frame gets its reaction, so both the belt (wheels)
    # and the frame (sprocket torque on the ankle) see the net torque motor - friction
    tau_ank, tau_w, in_air, ank_target = ankle_torques(tau_track - tau_loss_now, w_cmd, dq)
    # ankle drive: track Kd towards the sprocket speed in the air, passive damping (target 0) on the ground
    if (in_air != ank_air_prev).any():
        kd_a = np.where(in_air, rc.TRACK_KD, kd[:, ankles]).astype(np.float32)
        robot.set_dof_gains(stiffnesses=np.zeros_like(kd_a), dampings=kd_a, dof_indices=ankles)
        ank_air_prev[:] = in_air
    robot.set_dof_velocity_targets(np.where(in_air, ank_target, 0.0).astype(np.float32), dof_indices=ankles)
    if args.drive == 'implicit':
        # PhysX drive does the Kd loop, the feedforward effort is added as joint effort
        ff_w = -(np.clip(eff_cmd_now, -rc.MAX_FEEDFORWARD, rc.MAX_FEEDFORWARD) - tau_loss_now) * (R / RS) / NW
        eff = np.concatenate([tau_ank, np.repeat(ff_w, NW, axis=1)], 1).astype(np.float32)
        vt_w = np.zeros((NR, len(all_wheels)), np.float32)
        for j, w in enumerate(wheels):
            vt_w[:, j * NW:(j + 1) * NW] = (-w_cmd_now[:, j] * RS / R * ground_scale[:, j])[:, None]
        robot.set_dof_velocity_targets(vt_w, dof_indices=all_wheels)
    else:
        eff = np.concatenate([tau_ank, np.repeat(tau_w, NW, axis=1)], 1).astype(np.float32)
    robot.set_dof_efforts(eff, dof_indices=eff_idx)
    pt = np.zeros((NR, N), np.float32); vt = np.zeros((NR, N), np.float32)
    pt[:, hfe] = q_ref; vt[:, hfe] = dq_ref
    robot.set_dof_position_targets(pt[:, hfe], dof_indices=hfe)
    robot.set_dof_velocity_targets(vt[:, hfe], dof_indices=hfe)
    SimulationManager.step(steps=1)
    if args.frames and k >= n_settle and int((k - n_settle + 1) * args.dt * FPS) > int((k - n_settle) * args.dt * FPS):
        pos_, quat_, lin_, ang_ = get_state()
        w_, x_, y_, z_ = quat_[0]
        yaw_ = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        # copies: the arrays of the tensor API can be views of buffers reused on the next call
        frames['t'].append(t + args.dt); frames['pos'].append(pos_[0].copy()); frames['quat'].append(quat_[0].copy())
        frames['q'].append(robot.get_dof_positions().numpy()[0].copy())
        frames['vx'].append(math.cos(yaw_) * lin_[0, 0] + math.sin(yaw_) * lin_[0, 1]); frames['wz'].append(ang_[0, 2])
        frames['cmd'].append(ctrls[0].stick.copy()); frames['r2'].append(r2_on(t))
    if k >= n_settle and (k - n_settle) % log_every == 0:
        pos, quat, lin, ang = get_state()
        w, x, y, z = quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        c_, s_ = np.cos(yaw), np.sin(yaw)
        log['t'].append(t); log['pos'].append(pos - np.array([[args.spacing * r_, 0, 0] for r_ in range(NR)]))
        log['yaw'].append(yaw); log['roll'].append(roll); log['pitch'].append(pitch)
        log['vx'].append(c_ * lin[:, 0] + s_ * lin[:, 1]); log['vy'].append(-s_ * lin[:, 0] + c_ * lin[:, 1])
        log['wz'].append(ang[:, 2])
        log['w_track'].append(w_track); log['w_cmd'].append(w_cmd); log['eff_cmd'].append(eff_cmd)
        log['tau_track'].append(tau_track); log['q_hfe'].append(q[:, hfe]); log['q_ref'].append(q_ref)
        log['q_ankle'].append(q[:, ankles]); log['in_air'].append(in_air)
        tf = robot._physics_articulation_view.get_link_transforms().numpy().reshape(NR, len(links), 7)
        log['wheel_z'].append(np.stack([tf[:, wl, 2] for wl in wheel_links], 1))
        log['body_z'].append(tf[:, track_links, 2])
        # track (ankle) positions in the base frame, x forward
        base = tf[:, links.index('base_link'), :3]
        d_ = tf[:, track_links, :3] - base[:, None]
        log.setdefault('track_x', []).append(np.cos(yaw)[:, None] * d_[..., 0] + np.sin(yaw)[:, None] * d_[..., 1])
        log['roller_w'].append(np.stack([np.abs(dq[:, ro]).max(1) for ro in roller_of], 1))
    if k % int(round(10 / args.dt)) == 0:
        print(f'sim t {t:7.2f}  wall {time.time() - wall:6.1f} s', flush=True)

out = {k: np.array(v) for k, v in log.items()}
out['params'] = np.array([[p[k] if p[k] is not None else np.nan for k in GRID_KEYS] for p in PARAMS], float)
out['param_names'] = np.array(GRID_KEYS)
np.savez_compressed(args.out, **out)
if args.frames:
    np.savez_compressed(args.frames, dof_names=np.array(names), **{k: np.array(v) for k, v in frames.items()})
print(f'saved {args.out}: {len(log["t"])} samples x {NR} robots, wall {time.time() - wall:.0f} s', flush=True)
app.close()
