"""Replica of open_traquad.py (Isaac Lab) with the plain Isaac Sim API: same stance, drives, dt,
friction, track width and --demo command sequence. Prints measured vs commanded body velocities
per command segment and the wheel speed tracking.

usage: ./isaac.sh open_traquad_replica.py --usd <robot.usda> [--roller_damping d] [--wheel_kd 0.5]
"""
import argparse
from ankle_control import PASSIVE_DAMPING, SPROCKET_RADIUS, AnkleController
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--roller_friction', type=float, default=0.06,
                    help='dry (Coulomb) friction torque of the roller joints [Nm], always applied (0 = none; 0.06 = asset value)')
parser.add_argument('--roller_damping', type=float, default=3e-5)
parser.add_argument('--wheel_kd', type=float, default=0.5)
parser.add_argument('--track_width', type=float, default=0.395)
parser.add_argument('--ground_friction', type=float, default=1.0)
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
DT = 1.0 / 200.0
R = 0.015
DEMO = [(4.0, 0.3, 0.0), (4.0, 0.0, 0.8), (4.0, 0.3, 0.5), (4.0, -0.3, 0.0), (4.0, 0.0, -0.8), (2.0, 0.0, 0.0)]
STANCE = {'LF_HFE': 1.47, 'LH_HFE': -1.47, 'RF_HFE': -1.47, 'RH_HFE': 1.47,
          'body_left_F_ankle': 0.10, 'body_right_F_ankle': 0.10,
          'body_left_H_ankle': -0.10, 'body_right_H_ankle': -0.10}


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mat = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(args.ground_friction)
    m.CreateDynamicFrictionAttr().Set(args.ground_friction)
    UsdGeom.Xform.Define(stage, '/World/Ground')
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')
    plane.CreateAxisAttr().Set('Z')
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    stage_utils.add_reference_to_stage(usd_path=args.usd, path='/World/robot')
    robot_prim = stage.GetPrimAtPath('/World/robot')
    robot_prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    UsdGeom.XformCommonAPI(robot_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.30))
    await app_utils.update_app_async()
    roots = [p for p in stage.Traverse()
             if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    px = PhysxSchema.PhysxArticulationAPI.Apply(roots[0])
    px.CreateEnabledSelfCollisionsAttr().Set(False)
    px.CreateSolverPositionIterationCountAttr().Set(16)
    px.CreateSolverVelocityIterationCountAttr().Set(4)
    for p in stage.Traverse():   # max depenetration velocity 1.0 as in the Isaac Lab config
        if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr().Set(1.0)
    SimulationManager.setup_simulation(dt=DT, device='cpu')
    await app_utils.update_app_async()
    return Articulation(str(roots[0].GetPath()))


robot = app.run_coroutine(build())
app_utils.play()
app.update()

names = robot.dof_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
hfe = [idx[n] for n in ('LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE')]
ankles = [i for n, i in idx.items() if n.endswith('_ankle')]
left = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'LEFT' in n]
right = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'RIGHT' in n]
rollers = [i for n, i in idx.items() if '_roller_' in n]

kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
kp[hfe] = 100.0; kd[hfe] = 10.0; fmax[hfe] = 10.0
kd[ankles] = PASSIVE_DAMPING   # passive ankle; velocity control when the track is in the air (ankle_control)

kd[left + right] = args.wheel_kd; fmax[left + right] = 0.375   # track motor 1.5 N m per track
ankle_ctrl = AnkleController(robot, wheel_kd=args.wheel_kd, wheel_max=0.375)
kd[rollers] = args.roller_damping
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
robot.set_dof_max_efforts(fmax[None])
arm = np.zeros(N, np.float32); arm[left + right] = 0.001
robot.set_dof_armatures(arm[None])
q0 = np.zeros(N, np.float32)
for n, v in STANCE.items():
    q0[idx[n]] = v
robot.set_dof_positions(q0[None])
robot.set_dof_position_targets(q0[None])
if rollers:   # dry friction: a roller turns only above this torque
    tau = np.full((1, len(rollers)), args.roller_friction, np.float32)
    robot.set_dof_friction_properties(static_frictions=tau, dynamic_frictions=tau, dof_indices=rollers)
print(f'DOFs {N}: wheels {len(left)}+{len(right)} rollers {len(rollers)} | wheel Kd {args.wheel_kd} '
      f'| B {args.track_width} | ground mu {args.ground_friction}', flush=True)


def body_vel():
    _, q = robot.get_world_poses()
    lin, ang = robot.get_velocities()
    w, x, y, z = q.numpy()[0]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lin = lin.numpy()[0]
    return (math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1], float(ang.numpy()[0][2]),
            -math.sin(yaw) * lin[0] + math.cos(yaw) * lin[1])


SimulationManager.step(steps=int(1.0 / DT))   # settle 1 s
print(f'{"cmd v":>6} {"cmd w":>6} | {"meas v":>7} {"meas w":>7} | {"v %":>5} {"w %":>5} | {"|vy|":>6} | wheel speed % of cmd (L / R)')
for dur, v, w in DEMO:
    vl, vr = v - w * args.track_width / 2, v + w * args.track_width / 2
    tgt = np.zeros(N, np.float32)
    tgt[left] = -vl / R
    tgt[right] = vr / R
    robot.set_dof_velocity_targets(tgt[None])
    ankle_ctrl.set_speeds(vl, vr)
    acc = []
    for k in range(int(dur / DT)):
        SimulationManager.step(steps=1)
        if k * DT >= dur - 2.0:   # average over the last 2 s of the segment
            qd = robot.get_dof_velocities().numpy()[0]
            bv = body_vel()
            acc.append((bv[0], bv[1], -qd[left].mean() * R, qd[right].mean() * R, abs(bv[2])))
    a = np.array(acc).mean(axis=0)
    pv = f'{a[0] / v * 100:5.0f}' if v else '    -'
    pw = f'{a[1] / w * 100:5.0f}' if w else '    -'
    wl = f'{a[2] / vl * 100:4.0f}' if abs(vl) > 1e-6 else '   -'
    wr = f'{a[3] / vr * 100:4.0f}' if abs(vr) > 1e-6 else '   -'
    print(f'{v:+6.2f} {w:+6.2f} | {a[0]:+7.3f} {a[1]:+7.3f} | {pv} {pw} | {a[4]:6.3f} | {wl} / {wr}', flush=True)
app.close()
