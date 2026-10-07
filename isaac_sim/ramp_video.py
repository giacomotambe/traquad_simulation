"""Video of two robots parked across a real inclined ramp, with different roller dry friction.

The ground plane is tilted by --angle about x (downhill = -y); both robots face +x, so the slope is
lateral for them. Robot A (x = +0.6) uses --tau_a, robot B (x = -0.6) uses --tau_b. Mass scaled to --mass.
Frames saved as PNG named by simulated time.

usage: ./isaac.sh ramp_video.py --usd <robot.usda> --out <frames_dir> [--angle 20] [--tau_a 0] [--tau_b 0.018]
"""
import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--angle', type=float, default=20.0)
parser.add_argument('--tau_a', type=float, default=0.0)
parser.add_argument('--tau_b', type=float, default=0.018)
parser.add_argument('--roller_damping', type=float, default=1e-4)
parser.add_argument('--mass', type=float, default=7.8)
parser.add_argument('--duration', type=float, default=8.0)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'width': 1280, 'height': 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

SimulationManager.switch_physics_engine('physx')
os.makedirs(args.out, exist_ok=True)
DT = 1.0 / 200.0
TH = math.radians(args.angle)
NORMAL = np.array([0.0, -math.sin(TH), math.cos(TH)])      # ramp normal (rotation about x by +angle)
STANCE = {'LF_HFE': 1.47, 'LH_HFE': -1.47, 'RF_HFE': -1.47, 'RH_HFE': 1.47,
          'body_left_F_ankle': 0.10, 'body_right_F_ankle': 0.10,
          'body_left_H_ankle': -0.10, 'body_right_H_ankle': -0.10}
ROBOTS = [('/World/robot_a', 0.6, args.tau_a), ('/World/robot_b', -0.6, args.tau_b)]
H0 = 0.265   # base height above the ground at rest


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mat = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(1.0)
    m.CreateDynamicFrictionAttr().Set(1.0)
    ramp = UsdGeom.Xform.Define(stage, '/World/Ramp')
    UsdGeom.Xformable(ramp.GetPrim()).AddRotateXOp().Set(args.angle)
    plane = UsdGeom.Plane.Define(stage, '/World/Ramp/Plane')
    plane.CreateAxisAttr().Set('Z')
    plane.CreateWidthAttr().Set(30.0)
    plane.CreateLengthAttr().Set(30.0)
    plane.CreateDisplayColorAttr().Set([Gf.Vec3f(0.62, 0.6, 0.55)])
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    for i in range(-15, 16):   # 0.5 m grid lines on the ramp
        for name, size, pos in ((f'gx{i + 15}', (0.008, 15.0, 0.002), (i * 0.5, 0, 0.001)),
                                (f'gy{i + 15}', (15.0, 0.008, 0.002), (0, i * 0.5, 0.001))):
            c = UsdGeom.Cube.Define(stage, f'/World/Ramp/Grid/{name}')
            c.CreateSizeAttr().Set(1.0)
            c.CreateDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.38, 0.35)])
            xf = UsdGeom.Xformable(c.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr().Set(900.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr().Set(2500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
    arts = []
    for path, x, _ in ROBOTS:
        stage_utils.add_reference_to_stage(usd_path=args.usd, path=path)
        prim = stage.GetPrimAtPath(path)
        prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, 0.0, 2.0))   # out of the way until placed
        await app_utils.update_app_async()
        root = [p for p in stage.Traverse() if str(p.GetPath()).startswith(path) and p.HasAPI(UsdPhysics.ArticulationRootAPI)][0]
        px = PhysxSchema.PhysxArticulationAPI.Apply(root)
        px.CreateEnabledSelfCollisionsAttr().Set(False)
        px.CreateSolverPositionIterationCountAttr().Set(16)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        arts.append(str(root.GetPath()))
    SimulationManager.setup_simulation(dt=DT, device='cpu')
    await app_utils.update_app_async()
    return [Articulation(a) for a in arts]


robots = app.run_coroutine(build())
cam = RtxCamera('/World/Cam', tick_rate=0.0)
cam_usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath('/World/Cam'))
cam_usd.GetFocalLengthAttr().Set(18.0)
cam_usd.GetHorizontalApertureAttr().Set(20.955)
cam_usd.GetVerticalApertureAttr().Set(20.955 * 720 / 1280)
cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
sensor = CameraSensor(cam, resolution=(720, 1280), annotators=['rgb'])
app_utils.play(commit=True)
app.update()

quat = [math.cos(TH / 2), math.sin(TH / 2), 0.0, 0.0]          # robot tilted with the ramp (wxyz)
for robot, (path, x, tau) in zip(robots, ROBOTS):
    names = robot.dof_names
    N = len(names)
    idx = {n: i for i, n in enumerate(names)}
    hfe = [idx[n] for n in ('LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE')]
    ankles = [i for n, i in idx.items() if n.endswith('_ankle')]
    wheels = [i for n, i in idx.items() if n.startswith('joint_wheel_')]
    rollers = [i for n, i in idx.items() if '_roller_' in n]
    kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
    kp[hfe] = 100.0; kd[hfe] = 10.0; fmax[hfe] = 10.0
    kd[ankles] = 0.01   # passive ankles (asset value): parked, the tracks touch the ground
    kd[wheels] = 0.5; fmax[wheels] = 0.375   # track motor 1.5 N m per track
    kd[rollers] = args.roller_damping
    robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
    robot.set_dof_max_efforts(fmax[None])
    arm = np.zeros(N, np.float32); arm[wheels] = 0.001
    robot.set_dof_armatures(arm[None])
    if rollers:
        t_ = np.full((1, len(rollers)), tau, np.float32)
        robot.set_dof_friction_properties(static_frictions=t_, dynamic_frictions=t_, dof_indices=rollers)
    masses = robot.get_link_masses().numpy()
    k = args.mass / masses.sum()
    robot.set_link_masses(masses * k)
    robot.set_link_inertias(robot.get_link_inertias().numpy() * k)
    q0 = np.zeros(N, np.float32)
    for n, v in STANCE.items():
        q0[idx[n]] = v
    robot.set_dof_positions(q0[None])
    robot.set_dof_position_targets(q0[None])
    robot.set_dof_velocity_targets(np.zeros((1, N), np.float32))
    pos = np.array([x, 0.0, 0.0]) + NORMAL * H0
    robot.set_world_poses(positions=[pos.tolist()], orientations=[quat])
    robot.set_velocities(linear_velocities=np.zeros((1, 3), np.float32), angular_velocities=np.zeros((1, 3), np.float32))


def look_at(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mm = np.array([x, y, z]).T
    qw = math.sqrt(max(0.0, 1 + mm[0, 0] + mm[1, 1] + mm[2, 2])) / 2
    return [qw, (mm[2, 1] - mm[1, 2]) / (4 * qw), (mm[0, 2] - mm[2, 0]) / (4 * qw), (mm[1, 0] - mm[0, 1]) / (4 * qw)]


eye = np.array([3.2, 0.6, 1.4])
target = np.array([0.0, -0.6, -0.1])
cam.set_world_poses(positions=[eye.tolist()], orientations=[look_at(eye, target)])
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
except OSError:
    font = ImageFont.load_default()

t0 = SimulationManager.get_simulation_time()
start = [r.get_world_poses()[0].numpy()[0].copy() for r in robots]
n = 0
while SimulationManager.get_simulation_time() - t0 <= args.duration:
    app.update()
    data, _ = sensor.get_data('rgb')
    if data is None:
        continue
    t = SimulationManager.get_simulation_time() - t0
    slid = [float(-(r.get_world_poses()[0].numpy()[0] - s) @ np.array([0.0, math.cos(TH), math.sin(TH)]))
            for r, s in zip(robots, start)]
    img = Image.fromarray(np.asarray(data.numpy())[..., :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 84], fill=(255, 255, 255))
    d.text((20, 10), f'Isaac Sim - {args.angle:g} deg side slope - robot {args.mass:g} kg parked - t = {t:4.1f} s',
           fill=(20, 20, 20), font=font)
    d.text((20, 46), f'near: roller friction {ROBOTS[0][2]:g} Nm, slid {slid[0]:.2f} m      '
           f'far: roller friction {ROBOTS[1][2]:g} Nm, slid {slid[1]:.2f} m', fill=(20, 20, 20), font=font)
    img.save(os.path.join(args.out, f'{t:08.3f}.png'))
    n += 1
print('frames saved', n, flush=True)
app.close()
