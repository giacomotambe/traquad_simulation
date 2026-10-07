"""Lateral push test in Isaac Sim (training setup of open_traquad.py, robot standing still).

For each force a constant lateral force (base frame, +y, at the base link origin) is applied for
PUSH seconds, starting from the same settled state. Prints, per force: lateral displacement,
steady lateral velocity (last second) and yaw change.

usage: ./isaac.sh lateral_force.py --usd <robot.usda> [--roller_damping d] [--forces 2 5 10 20 40]
"""
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--roller_friction', type=float, default=0.06,
                    help='dry (Coulomb) friction torque of the roller joints [Nm], always applied (0 = none; 0.06 = asset value)')
parser.add_argument('--mu', type=float, default=None,
                    help='friction of ground and robot, same material on both (default: ground 1.0, robot default 0.5, '
                         'i.e. 0.75 effective as in open_traquad.py)')
parser.add_argument('--roller_damping', type=float, default=1e-4)
parser.add_argument('--forces', type=float, nargs='+', default=[2.0, 5.0, 10.0, 20.0, 40.0])
parser.add_argument('--push', type=float, default=3.0, help='push duration [s]')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import warp as wp  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

SimulationManager.switch_physics_engine('physx')
DT = 1.0 / 200.0
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
    m.CreateStaticFrictionAttr().Set(1.0 if args.mu is None else args.mu)
    m.CreateDynamicFrictionAttr().Set(1.0 if args.mu is None else args.mu)
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
    if args.mu is not None:   # same material on every collider of the robot: wheel-ground friction = mu
        for p in stage.Traverse():
            if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
    roots = [p for p in stage.Traverse()
             if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    px = PhysxSchema.PhysxArticulationAPI.Apply(roots[0])
    px.CreateEnabledSelfCollisionsAttr().Set(False)
    px.CreateSolverPositionIterationCountAttr().Set(16)
    px.CreateSolverVelocityIterationCountAttr().Set(4)
    for p in stage.Traverse():
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
wheels = [i for n, i in idx.items() if n.startswith('joint_wheel_')]
rollers = [i for n, i in idx.items() if '_roller_' in n]
kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
kp[hfe] = 100.0; kd[hfe] = 10.0; fmax[hfe] = 5.0
kd[ankles] = 0.05   # passive ankles, as in Gazebo and in the asset
kd[wheels] = 0.5; fmax[wheels] = 10.0          # wheels held at zero speed (velocity drive)
kd[rollers] = args.roller_damping
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
robot.set_dof_max_efforts(fmax[None])
arm = np.zeros(N, np.float32); arm[wheels] = 0.001
robot.set_dof_armatures(arm[None])
q0 = np.zeros(N, np.float32)
for n, v in STANCE.items():
    q0[idx[n]] = v
robot.set_dof_positions(q0[None])
robot.set_dof_position_targets(q0[None])
robot.set_dof_velocity_targets(np.zeros((1, N), np.float32))
if rollers:   # dry friction: a roller turns only above this torque
    tau = np.full((1, len(rollers)), args.roller_friction, np.float32)
    robot.set_dof_friction_properties(static_frictions=tau, dynamic_frictions=tau, dof_indices=rollers)

view = robot._physics_articulation_view
base = robot.link_names.index('base_link')
n_links = len(robot.link_names)
indices = wp.array([0], dtype=wp.int32, device='cpu')


def pose():
    p, q = robot.get_world_poses()
    w, x, y, z = q.numpy()[0]
    return p.numpy()[0].copy(), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


SimulationManager.step(steps=int(2.0 / DT))      # settle
p_rest, q_rest = [a.numpy().copy() for a in robot.get_world_poses()]
dof_rest = robot.get_dof_positions().numpy().copy()
print(f'DOFs {N} rollers {len(rollers)} damping {args.roller_damping:g} friction {args.roller_friction:g} | rest base z {p_rest[0][2]:.3f}', flush=True)

for F in args.forces:
    # restore the settled state
    robot.set_world_poses(positions=p_rest, orientations=q_rest)
    robot.set_velocities(linear_velocities=np.zeros((1, 3), np.float32), angular_velocities=np.zeros((1, 3), np.float32))
    robot.set_dof_positions(dof_rest)
    robot.set_dof_velocities(np.zeros((1, N), np.float32))
    SimulationManager.step(steps=int(1.0 / DT))
    p0, yaw0 = pose()
    force = np.zeros((1, n_links, 3), np.float32)
    force[0, base, 1] = F                         # +y in the base frame
    f_wp = wp.array(force, dtype=wp.float32, device='cpu')
    vy, w_roll = [], []
    for k in range(int(args.push / DT)):
        view.apply_forces_and_torques_at_position(f_wp, None, None, indices, False)
        SimulationManager.step(steps=1)
        if k * DT >= args.push - 1.0:
            lin, _ = robot.get_velocities()
            p, yaw = pose()
            lin = lin.numpy()[0]
            vy.append(-math.sin(yaw) * lin[0] + math.cos(yaw) * lin[1])
            w_roll.append(np.abs(robot.get_dof_velocities().numpy()[0][rollers]).mean() if rollers else 0.0)
    p1, yaw1 = pose()
    d = p1 - p0
    lat = -math.sin(yaw0) * d[0] + math.cos(yaw0) * d[1]
    print(f'RES d={args.roller_damping:g} tau={args.roller_friction:g} F={F:g} lateral_disp {lat:+.4f} m | vy_steady {np.mean(vy):+.4f} m/s '
          f'| roller_vel {np.mean(w_roll):.2f} rad/s | yaw {math.degrees(yaw1 - yaw0):+.2f} deg | z {p1[2]:.3f} | mu {args.mu}', flush=True)
app.close()
