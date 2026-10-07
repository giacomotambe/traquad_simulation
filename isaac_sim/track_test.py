"""Traquad standard maneuver in Isaac Sim (PhysX), same setup and metrics as the Gazebo tests.

Maneuver (simulated time): 3 s settle, 8 s rotation in place (w = 0.5 rad/s), 2 s stop,
10 s turn (v = 0.2 m/s, w = 0.5 rad/s), 2 s stop, 5 s straight (v = 0.2 m/s). Logs the base pose to CSV and prints
the yaw ratio, the lateral drift and the final pose error vs the ideal unicycle, plus
- jitter: standard deviation of the vertical acceleration of the base in the straight phase (after 1 s)
- wheel torque demand: mean |torque| of the driven wheels over their limit, and time at the limit, per phase
- stability (whole run after the first 0.5 s): 99th percentile of |base vertical acceleration|, max |roll/pitch
  rate| of the base, max |velocity| of rollers and ankles, NaN check.
Works with both track models: with the cylinder model only the driven wheels (wheel 2, not mimic followers) are
commanded. Wheel gain and torque limit are read from the USD (written by finalize_usd.py) unless overridden.

usage: ./isaac.sh track_test.py --usd <robot.usda> --out <csv> [--damping d] [--mu 0.5] [--mu_dyn 0.45] [--info]
"""
import argparse
from ankle_control import PASSIVE_DAMPING, AnkleController
import csv
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--damping', type=float, default=1e-4, help='roller joint damping [Nms/rad]')
parser.add_argument('--roller_friction', type=float, default=None,
                    help='roller joint dry friction [Nm] (default: from the USD, 0.06)')
parser.add_argument('--mu', type=float, default=0.75,
                    help='isotropic (static) friction of ground and robot (0.75 = effective value of open_traquad.py)')
parser.add_argument('--mu_dyn', type=float, default=None, help='dynamic friction (default: same as --mu)')
parser.add_argument('--wheel_kd', type=float, default=None, help='driven wheel velocity gain (default: from the USD)')
parser.add_argument('--wheel_max', type=float, default=None, help='driven wheel torque limit (default: from the USD)')
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--friction_corr', type=float, default=None,
                    help='PhysX friction correlation distance [m] (default 0.025): contacts closer than this share '
                         'one friction anchor')
parser.add_argument('--info', action='store_true', help='only check settle state and forward motion')
parser.add_argument('--hfe', type=float, default=1.47,
                    help='HFE stance [rad] (1.47: RL tasks and open_traquad.py; 1.13: earlier tests); ankles start flat')
parser.add_argument('--no_coupling', action='store_true', help='ankles always passive (no velocity control in the air)')
parser.add_argument('--ankle_range', type=float, default=None, help='ankle end stops +- this around the flat pose of the '
                    'default stance (0.1008) [rad] (default: from the USD, +-0.5236)')
parser.add_argument('--ankle_kd', type=float, default=None, help='passive ankle damping [N m s/rad] (default: ankle_control)')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

SimulationManager.switch_physics_engine('physx')
print('ENGINE', SimulationManager.get_active_physics_engine(), flush=True)

R, B = 0.015, 0.395
HFE_TARGET = {'LF_HFE': args.hfe, 'LH_HFE': -args.hfe, 'RF_HFE': -args.hfe, 'RH_HFE': args.hfe}
ANKLE_FLAT = math.pi / 2 - args.hfe   # |HFE| + |ankle| = pi/2: track flat on the ground
PHASES = [('settle', 0.0, 0.0, 3.0), ('yaw', 0.0, 0.5, 8.0), ('stop1', 0.0, 0.0, 2.0),
          ('curve', 0.2, 0.5, 10.0), ('stop2', 0.0, 0.0, 2.0), ('straight', 0.2, 0.0, 5.0)]
MU_DYN = args.mu if args.mu_dyn is None else args.mu_dyn
# exact (analytic) cylinder colliders unless this is set
print('collisionApproximateCylinders:', carb.settings.get_settings().get('/physics/collisionApproximateCylinders'),
      flush=True)


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # physics material (isotropic)
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterial')
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(args.mu)
    m.CreateDynamicFrictionAttr().Set(MU_DYN)
    m.CreateRestitutionAttr().Set(0.0)
    # ground: infinite static plane at z = 0
    UsdGeom.Xform.Define(stage, '/World/Ground')
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')
    plane.CreateAxisAttr().Set('Z')
    plane.CreateWidthAttr().Set(100.0)
    plane.CreateLengthAttr().Set(100.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    # robot
    stage_utils.add_reference_to_stage(usd_path=args.usd, path='/World/robot')
    robot_prim = stage.GetPrimAtPath('/World/robot')
    robot_prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    UsdGeom.XformCommonAPI(robot_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.30))
    await app_utils.update_app_async()
    # bind the same physics material to every collider of the robot
    n_col = 0
    for p in stage.Traverse():
        if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
            n_col += 1
    print('robot colliders with material:', n_col, flush=True)
    SimulationManager.setup_simulation(dt=args.dt, device='cpu')
    for ps in stage.Traverse():
        if ps.IsA(UsdPhysics.Scene):
            px = PhysxSchema.PhysxSceneAPI.Apply(ps)
            px.CreateSolverTypeAttr().Set('TGS')
            if args.friction_corr is not None:
                px.CreateFrictionCorrelationDistanceAttr().Set(args.friction_corr)
            ps.GetAttribute('physxScene:enableGPUDynamics').Set(False) if ps.GetAttribute(
                'physxScene:enableGPUDynamics') else None
    await app_utils.update_app_async()
    roots = [str(p.GetPath()) for p in stage.Traverse()
             if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print('articulation roots:', roots, flush=True)
    # no self-collisions inside the robot (as in Gazebo); PhysX enables them by default
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(roots[0])).CreateEnabledSelfCollisionsAttr().Set(False)
    # wheel drives written in the USD; mimic followers (cylinder model) have no drive
    global FOLLOWERS, USD_WHEEL
    FOLLOWERS, USD_WHEEL = set(), {}
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith('/World/robot') or not p.GetName().startswith('joint_wheel_'):
            continue
        rel = p.GetRelationship('newton:mimicJoint')
        if rel and rel.GetTargets():
            FOLLOWERS.add(p.GetName())
        elif p.HasAPI(UsdPhysics.DriveAPI, 'angular'):
            d = UsdPhysics.DriveAPI(p, 'angular')
            USD_WHEEL = {'kd': d.GetDampingAttr().Get() * 180.0 / math.pi, 'max': d.GetMaxForceAttr().Get()}
    return Articulation(roots[0])


robot = app.run_coroutine(build())
app_utils.play()
app.update()

names = robot.dof_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
hfe = [idx[n] for n in HFE_TARGET]
wheels_l = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'LEFT' in n and n not in FOLLOWERS]
wheels_r = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'RIGHT' in n and n not in FOLLOWERS]
WHEEL_KD = USD_WHEEL['kd'] if args.wheel_kd is None else args.wheel_kd
WHEEL_MAX = USD_WHEEL['max'] if args.wheel_max is None else args.wheel_max
ankles = [i for n, i in idx.items() if n.endswith('_ankle')]
ANK_NAMES = [n for n in idx if n.endswith('_ankle')]
# driven wheels of each track, in the order of ANK_NAMES (body_left_H -> LEFT_H, ...)
TRACK_WHEELS = [[i for m, i in idx.items() if m.startswith('joint_wheel_') and m.endswith(n.split('_')[1].upper() + '_'
                 + n.split('_')[2]) and m not in FOLLOWERS] for n in ANK_NAMES]
ANK_FLAT = np.array([(math.pi / 2 - args.hfe) * (1 if '_F_' in n else -1) for n in ANK_NAMES])
rollers = [i for n, i in idx.items() if '_roller_' in n]
print(f'DOFs {N}: hfe {len(hfe)} driven wheels {len(wheels_l)}+{len(wheels_r)} mimic wheels {len(FOLLOWERS)} '
      f'ankles {len(ankles)} rollers {len(rollers)} | wheel kd {WHEEL_KD:g} max {WHEEL_MAX:g} Nm | '
      f'mu {args.mu}/{MU_DYN} dt {args.dt}', flush=True)

kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
kp[hfe] = 100.0; kd[hfe] = 10.0; fmax[hfe] = 10.0
kd[wheels_l + wheels_r] = WHEEL_KD; fmax[wheels_l + wheels_r] = WHEEL_MAX
kd[ankles] = PASSIVE_DAMPING if args.ankle_kd is None else args.ankle_kd   # passive ankle (velocity control in the air below)
ankle_ctrl = AnkleController(robot, enabled=not args.no_coupling)
if args.ankle_range is not None:
    lo_, hi_ = (x.numpy() for x in robot.get_dof_limits())
    for n_, i_ in idx.items():
        if n_.endswith('_ankle'):
            c_ = 0.1008 if '_F_' in n_ else -0.1008
            lo_[0, i_], hi_[0, i_] = c_ - args.ankle_range, c_ + args.ankle_range
    robot.set_dof_limits(lo_, hi_)
kd[rollers] = args.damping
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
if rollers and args.roller_friction is not None:
    tau = np.full((1, len(rollers)), args.roller_friction, np.float32)
    robot.set_dof_friction_properties(static_frictions=tau, dynamic_frictions=tau, dof_indices=rollers)
robot.set_dof_max_efforts(fmax[None])
pos_t = np.zeros(N, np.float32)
for n, q in HFE_TARGET.items():
    pos_t[idx[n]] = q
robot.set_dof_position_targets(pos_t[None])
# initial state as it ends up in Gazebo: legs at their target, tracks flat (ankle at the flat pose)
q0 = pos_t.copy()
for n in ('body_left_F_ankle', 'body_right_F_ankle'):
    q0[idx[n]] = ANKLE_FLAT
for n in ('body_left_H_ankle', 'body_right_H_ankle'):
    q0[idx[n]] = -ANKLE_FLAT
robot.set_dof_positions(q0[None])


def state():
    p, q = robot.get_world_poses()
    lin, ang = robot.get_velocities()
    p, q, lin, ang = p.numpy()[0], q.numpy()[0], lin.numpy()[0], ang.numpy()[0]
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    return p, yaw, c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1], ang[2]


track_speed = [0.0, 0.0]   # belt speed of the left and right tracks, for the ankle controller
wheel_tgt = np.zeros(N, np.float32)   # wheel velocity targets (for the torque estimate)


def send(v, w):
    vel = np.zeros(N, np.float32)
    vel[wheels_l] = -(v - w * B / 2) / R        # URDF convention: left wheel axis = -y
    vel[wheels_r] = (v + w * B / 2) / R
    track_speed[:] = [v - w * B / 2, v + w * B / 2]
    ankle_ctrl.set_speeds(*track_speed)
    wheel_tgt[:] = vel
    robot.set_dof_velocity_targets(vel[None])


if args.info:
    send(0, 0)
    print('self collisions:', robot.get_enabled_self_collisions().numpy().ravel().tolist() if hasattr(
        robot.get_enabled_self_collisions(), 'numpy') else robot.get_enabled_self_collisions())
    tf_ln = robot.link_names
    for k in range(20):
        SimulationManager.step(steps=int(0.05 / args.dt))
        tf = robot._physics_articulation_view.get_link_transforms().numpy()[0]
        p, yaw, vx, vy, wz = state()
        zs = [tf[tf_ln.index(f'wheel_2_{s}'), 2] for s in ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')]
        print(f't {0.05 * (k + 1):.2f} base z {p[2]:.4f} x {p[0]:+.4f} yaw {math.degrees(yaw):+.2f} '
              f'wheel2 z ' + ' '.join(f'{z:.4f}' for z in zs))
    SimulationManager.step(steps=int(2.0 / args.dt))
    p, yaw, vx, vy, wz = state()
    q = robot.get_dof_positions().numpy()[0]
    print('base', np.round(p, 4), 'yaw', round(math.degrees(yaw), 2))
    print('HFE', {n: round(float(q[idx[n]]), 3) for n in HFE_TARGET})
    print('ankles', {names[i]: round(float(q[i]), 3) for i in ankles})
    tf = robot._physics_articulation_view.get_link_transforms().numpy()[0]   # (L, 7) xyz + quat
    ln = robot.link_names
    for side in ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H'):
        z1, z3 = tf[ln.index(f'wheel_1_{side}'), 2], tf[ln.index(f'wheel_3_{side}'), 2]
        x1, x3 = tf[ln.index(f'wheel_1_{side}'), 0], tf[ln.index(f'wheel_3_{side}'), 0]
        print(f'track {side}: wheel1 z {z1:.4f} wheel3 z {z3:.4f} pitch {math.degrees(math.atan2(z1 - z3, abs(x1 - x3))):+.1f} deg')
    send(0.2, 0.0)
    SimulationManager.step(steps=int(2.0 / args.dt))
    p2, *_ = state()
    print('forward 2 s: dx', round(float(p2[0] - p[0]), 3), 'dy', round(float(p2[1] - p[1]), 3))
    app.close()
    raise SystemExit

driven = wheels_l + wheels_r
rows, t, log_every = [], 0.0, max(1, int(round(0.05 / args.dt)))
torque = {}        # phase -> list of |torque| / limit of the driven wheels, every step
vz_straight = []   # base vertical velocity in the straight phase, every step
stab = {'vz': [], 'wxy': [], 'roller': [], 'ankle': [], 'nan': False, 'hfe_dev': [], 'at_stop': [], 'tau_ankle': [],
        'tau_hfe': []}
lo_lim, hi_lim = (x.numpy()[0] for x in robot.get_dof_limits())
for name, v, w, dur in PHASES:
    send(v, w)
    for k in range(int(round(dur / args.dt))):
        SimulationManager.step(steps=1)
        t += args.dt
        tau = robot.get_dof_projected_joint_forces().numpy()[0, driven]
        torque.setdefault(name, []).append(np.abs(tau) / WHEEL_MAX)
        lin, ang = (x.numpy()[0] for x in robot.get_velocities())
        if name == 'straight' and k * args.dt >= 1.0:
            vz_straight.append(lin[2])
        if t >= 0.5:
            dv = robot.get_dof_velocities().numpy()[0]
            stab['nan'] |= bool(np.isnan(lin).any() or np.isnan(dv).any())
            stab['vz'].append(lin[2]); stab['wxy'].append(np.abs(ang[:2]).max())
            stab['roller'].append(np.abs(dv[rollers]).max() if rollers else 0.0)
            stab['ankle'].append(np.abs(dv[ankles]).max())
            qq = robot.get_dof_positions().numpy()[0]
            stab['hfe_dev'].append(np.abs(qq[hfe] - pos_t[hfe]).max())
            stab['at_stop'].append(np.any(np.minimum(qq[ankles] - lo_lim[ankles], hi_lim[ankles] - qq[ankles]) < 0.02))
            # drive torques from the drive laws (the projected joint forces of the tensor API read zero)
            t_ank = getattr(ankle_ctrl, 'last_tau', np.zeros(1))   # controller torque (passive damping excluded)
            t_hfe = np.clip(kp[hfe] * (pos_t[hfe] - qq[hfe]) - kd[hfe] * dv[hfe], -fmax[hfe], fmax[hfe])
            stab['tau_ankle'].append(np.abs(t_ank).max()); stab['tau_hfe'].append(np.abs(t_hfe).max())
            stab.setdefault('phase', {}).setdefault(name, []).append(
                np.concatenate([qq[ankles] - ANK_FLAT,
                                np.abs(t_hfe) >= fmax[hfe] - 1e-3,
                                [np.clip(kd[w_] * (wheel_tgt[w_] - dv[w_]), -fmax[w_], fmax[w_]).sum() for w_ in TRACK_WHEELS],
                                np.abs(t_hfe)]))
        if k % log_every == 0:
            p, yaw, vx, vy, wz = state()
            rows.append([round(t, 3), name, v, w, p[0], p[1], p[2], yaw, vx, vy, wz])
with open(args.out, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['t', 'phase', 'v_cmd', 'w_cmd', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'wz'])
    wr.writerows(rows)

# metrics (same as Gazebo)
a = np.array([[r[0], r[4], r[5], r[7], r[9], r[10]] for r in rows])
ph = np.array([r[1] for r in rows])
m1 = (ph == 'yaw') & (a[:, 0] > 5.0)
m2 = (ph == 'curve') & (a[:, 0] > 15.0)
print(f'RESULT yaw_in_place {a[m1, 5].mean() / 0.5 * 100:.0f}% | yaw_turn {a[m2, 5].mean() / 0.5 * 100:.0f}% '
      f'| drift_turn {np.abs(a[m2, 4]).mean():.3f} m/s | z {rows[-1][6]:.3f}', flush=True)
ms = (ph == 'straight') & (a[:, 0] > a[ph == 'straight', 0].min() + 1.0)
az = np.diff(np.array(vz_straight)) / args.dt
print(f'RESULT straight_speed {a[ms, 0].size and np.hypot(np.diff(a[ms, 1]), np.diff(a[ms, 2])).sum() / (a[ms, 0][-1] - a[ms, 0][0]):.3f} m/s '
      f'(cmd 0.2) | jitter std(base z acc) {az.std():.3f} m/s^2', flush=True)
az_all = np.abs(np.diff(np.array(stab['vz'])) / args.dt)
print(f"RESULT legs: max |HFE - target| {math.degrees(max(stab['hfe_dev'])):.2f} deg | time with an ankle at a stop "
      f"{100 * np.mean(stab['at_stop']):.0f}% | max |ankle torque| {max(stab['tau_ankle']):.3f} N m | max |HFE torque| "
      f"{max(stab['tau_hfe']):.2f} N m", flush=True)
for ph, arr in stab['phase'].items():
    a_ = np.array(arr)
    print(f"PHASE {ph:8s} ankle from flat mean [deg] ({' '.join(n.replace('body_', '').replace('_ankle', '') for n in ANK_NAMES)}) "
          + ' '.join(f'{math.degrees(x):+5.1f}' for x in a_[:, :4].mean(0))
          + ' | min ' + ' '.join(f'{math.degrees(x):+5.1f}' for x in a_[:, :4].min(0))
          + ' | max ' + ' '.join(f'{math.degrees(x):+5.1f}' for x in a_[:, :4].max(0))
          + f" | HFE saturated {100 * a_[:, 4:8].any(1).mean():.0f}% of the time"
          + ' | track motor torque mean / max [N m] ' + ' '.join(f'{np.abs(a_[:, 8 + j]).mean():.2f}/{np.abs(a_[:, 8 + j]).max():.1f}' for j in range(4))
          + f' | HFE torque mean {a_[:, 12:16].mean():.2f} N m', flush=True)
print(f"RESULT stability: p99 |base z acc| {np.percentile(az_all, 99):.2f} m/s^2 | max |roll/pitch rate| "
      f"{max(stab['wxy']):.3f} rad/s | max |roller vel| {max(stab['roller']):.0f} rad/s | max |ankle vel| "
      f"{max(stab['ankle']):.2f} rad/s | NaN {stab['nan']}", flush=True)
print('RESULT wheel torque demand (mean |tau|/limit, time at >= 95% of the limit): ' + ' | '.join(
    f'{p} {np.mean(torque[p]):.2f} {np.mean(np.array(torque[p]) >= 0.95) * 100:.0f}%'
    for p in ('settle', 'yaw', 'curve', 'straight')), flush=True)


def ideal(p, v, w, T):
    x, y, th = p
    return x + v / w * (math.sin(th + w * T) - math.sin(th)), y - v / w * (math.cos(th + w * T) - math.cos(th)), th + w * T


def at(tq):
    i = int(np.argmin(np.abs(a[:, 0] - tq)))
    return a[i, 1], a[i, 2], a[i, 3]


yaw_unwrapped = np.unwrap(a[:, 3])
def yaw_at(tq):
    return yaw_unwrapped[int(np.argmin(np.abs(a[:, 0] - tq)))]


s0 = at(3.0); s0 = (s0[0], s0[1], yaw_at(3.0))
e1 = at(13.0); e1 = (e1[0], e1[1], yaw_at(13.0))
e2 = at(25.0); e2 = (e2[0], e2[1], yaw_at(25.0))
d2 = ideal(e1, 0.2, 0.5, 10.0)
dt_ = ideal(ideal(s0, 1e-9, 0.5, 8.0), 0.2, 0.5, 10.0)
print(f'RESULT turn_pos_err {math.hypot(e2[0] - d2[0], e2[1] - d2[1]):.3f} m | '
      f'total_pos_err {math.hypot(e2[0] - dt_[0], e2[1] - dt_[1]):.3f} m | '
      f'total_yaw_err {math.degrees(e2[2] - dt_[2]):+.1f} deg (of {math.degrees(dt_[2] - s0[2]):.0f})', flush=True)
app.close()
