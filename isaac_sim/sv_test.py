"""Track model with surface velocity: same maneuver and metrics as track_test.py, the belt is a PhysX surface velocity.

Each track is one contact box on the track body (no wheels); PhysxSurfaceVelocityAPI on the track body imposes the
belt speed along the track through PhysX contact modification. The USD is not in the repository: it was built from
the cylinder URDF by removing the wheel links (their mass added to the track body) and adding to each track body a
box collider under wheels 1-3 (0.045 x 0.129 x 0.03 m); --track_dirs gives the track direction in each body frame.

Findings (Isaac Sim 6.1): PhysX reads the surface velocity only when the simulation starts (later USD changes, even
with an app update, are ignored: use --static), it is not applied with GPU dynamics, and the robot still does not
turn (9% of the yaw rate in place, 0% in a curve). See the README.

Maneuver (simulated time): 3 s settle, 8 s rotation in place (w = 0.5 rad/s), 2 s stop, 10 s turn (v = 0.2 m/s,
w = 0.5 rad/s), 2 s stop, 5 s straight (v = 0.2 m/s).

usage: ./isaac.sh sv_test.py --usd <robot.usda> --track_dirs <json> --out <csv> [--mu 0.5] [--mu_dyn 0.45]
       [--dt 0.001] [--device cpu|cuda:0] [--sign 1]
"""
import argparse
import csv
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--track_dirs', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--mu', type=float, default=0.5)
parser.add_argument('--mu_dyn', type=float, default=None)
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--device', default='cpu', help='cpu or cuda:0 (GPU dynamics)')
parser.add_argument('--sign', type=float, default=1.0, help='sign of the surface velocity along the track direction')
parser.add_argument('--static', type=float, nargs=2, default=None, metavar=('V', 'W'),
                    help='constant command (v [m/s], w [rad/s]) set before play: PhysX reads the surface velocity '
                         'only when the simulation starts, later USD changes are ignored')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

SimulationManager.switch_physics_engine('physx')
B = 0.395
MU_DYN = args.mu if args.mu_dyn is None else args.mu_dyn
HFE_TARGET = {'LF_HFE': 1.13, 'LH_HFE': -1.13, 'RF_HFE': -1.13, 'RH_HFE': 1.13}
PHASES = [('settle', 0.0, 0.0, 3.0), ('yaw', 0.0, 0.5, 8.0), ('stop1', 0.0, 0.0, 2.0),
          ('curve', 0.2, 0.5, 10.0), ('stop2', 0.0, 0.0, 2.0), ('straight', 0.2, 0.0, 5.0)]
TRACKS = json.load(open(args.track_dirs))   # side -> {body, dir}
SV = {}                                     # side -> PhysxSurfaceVelocityAPI


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterial')
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(args.mu)
    m.CreateDynamicFrictionAttr().Set(MU_DYN)
    m.CreateRestitutionAttr().Set(0.0)
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')
    plane.CreateAxisAttr().Set('Z')
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    stage_utils.add_reference_to_stage(usd_path=args.usd, path='/World/robot')
    robot_prim = stage.GetPrimAtPath('/World/robot')
    robot_prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    UsdGeom.XformCommonAPI(robot_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.30))
    await app_utils.update_app_async()
    for p in stage.Traverse():
        path = str(p.GetPath())
        if path.startswith('/World/robot') and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
        if path.startswith('/World/robot') and p.HasAPI(UsdPhysics.RigidBodyAPI):
            for side, t in TRACKS.items():
                if p.GetName() == t['body']:
                    sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(p)
                    sv.CreateSurfaceVelocityEnabledAttr().Set(True)
                    sv.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
                    v0, w0 = args.static or (0.0, 0.0)
                    u0 = args.sign * (v0 - w0 * B / 2 if 'LEFT' in side else v0 + w0 * B / 2)
                    sv.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(*[u0 * c for c in t['dir']]))
                    SV[side] = sv
    print('surface velocity on', sorted(SV), flush=True)
    SimulationManager.setup_simulation(dt=args.dt, device=args.device)
    gpu = args.device != 'cpu'
    for ps in stage.Traverse():
        if ps.IsA(UsdPhysics.Scene):
            px = PhysxSchema.PhysxSceneAPI.Apply(ps)
            px.CreateSolverTypeAttr().Set('TGS')
            px.CreateEnableGPUDynamicsAttr().Set(gpu)
            px.CreateBroadphaseTypeAttr().Set('GPU' if gpu else 'MBP')
    await app_utils.update_app_async()
    roots = [str(p.GetPath()) for p in stage.Traverse()
             if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(roots[0])).CreateEnabledSelfCollisionsAttr().Set(False)
    return Articulation(roots[0])


robot = app.run_coroutine(build())
app_utils.play()
app.update()

names = robot.dof_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
print(f'DOFs {N}: {names} | mu {args.mu}/{MU_DYN} dt {args.dt} device {args.device}', flush=True)
kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32)
for n in HFE_TARGET:
    kp[idx[n]] = 100.0; kd[idx[n]] = 10.0
for n in idx:
    if n.endswith('_ankle'):
        kd[idx[n]] = 0.05
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
pos_t = np.zeros(N, np.float32)
for n, q in HFE_TARGET.items():
    pos_t[idx[n]] = q
robot.set_dof_position_targets(pos_t[None])
q0 = pos_t.copy()
for n in idx:
    if n.endswith('F_ankle'):
        q0[idx[n]] = 0.441
    elif n.endswith('H_ankle'):
        q0[idx[n]] = -0.441
robot.set_dof_positions(q0[None])


def state():
    p, q = robot.get_world_poses()
    lin, ang = robot.get_velocities()
    p, q, lin, ang = p.numpy()[0], q.numpy()[0], lin.numpy()[0], ang.numpy()[0]
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    return p, yaw, c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1], ang[2]


def send(v, w):
    """Belt speed of each track = speed of the robot over the ground on that side (skid-steer kinematics)."""
    if args.static is not None:
        return
    for side, sv in SV.items():
        u = v - w * B / 2 if 'LEFT' in side else v + w * B / 2
        d = TRACKS[side]['dir']
        sv.GetSurfaceVelocityAttr().Set(Gf.Vec3f(*[args.sign * u * c for c in d]))


rows, t, log_every, vz_straight = [], 0.0, max(1, int(round(0.05 / args.dt))), []
for name, v, w, dur in PHASES:
    send(v, w)
    for k in range(int(round(dur / args.dt))):
        SimulationManager.step(steps=1)
        t += args.dt
        if name == 'straight' and k * args.dt >= 1.0:
            vz_straight.append(robot.get_velocities()[0].numpy()[0, 2])
        if k % log_every == 0:
            p, yaw, vx, vy, wz = state()
            rows.append([round(t, 3), name, v, w, p[0], p[1], p[2], yaw, vx, vy, wz])
with open(args.out, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['t', 'phase', 'v_cmd', 'w_cmd', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'wz'])
    wr.writerows(rows)

a = np.array([[r[0], r[4], r[5], r[7], r[8], r[9], r[10]] for r in rows])
if args.static is not None:
    mm = a[:, 0] > 5.0
    print(f'RESULT static cmd v {args.static[0]:+.2f} w {args.static[1]:+.2f} | measured v {a[mm, 4].mean():+.3f} '
          f'w {a[mm, 6].mean():+.3f} ({a[mm, 6].mean() / args.static[1] * 100 if args.static[1] else 0:.0f}% of w) '
          f'| lateral {a[mm, 5].mean():+.3f} m/s', flush=True)
    app.close()
    raise SystemExit
ph = np.array([r[1] for r in rows])
m1 = (ph == 'yaw') & (a[:, 0] > 5.0)
m2 = (ph == 'curve') & (a[:, 0] > 15.0)
ms = (ph == 'straight') & (a[:, 0] > a[ph == 'straight', 0].min() + 1.0)
az = np.diff(np.array(vz_straight)) / args.dt
print(f'RESULT yaw_in_place {a[m1, 6].mean() / 0.5 * 100:.0f}% | yaw_turn {a[m2, 6].mean() / 0.5 * 100:.0f}% '
      f'| drift_turn {np.abs(a[m2, 5]).mean():.3f} m/s | z {rows[-1][6]:.3f}', flush=True)
print(f'RESULT straight vx {a[ms, 4].mean():+.3f} m/s (cmd +0.2) | turn vx {a[m2, 4].mean():+.3f} m/s (cmd +0.2) '
      f'| jitter std(base z acc) {az.std():.3f} m/s^2', flush=True)
app.close()
