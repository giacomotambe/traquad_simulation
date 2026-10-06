"""Render a video of the open_traquad.py setup in Isaac Sim with a follow camera.

Frames are saved as PNG named by simulated time, with the command and the measured body velocities
written on them. Works with both track models: with the cylinder model only the driven wheels (wheel 2, not
the mimic followers) are commanded. Track motor as in open_traquad.py: 40 Nm and 2 Nm s/rad per track, split over
its driven wheels. --csv logs command and measured velocities of every frame.
usage: ./isaac.sh video_replica.py --usd <robot.usda> --out <frames_dir> [--roller_damping 1e-4] [--csv log.csv]
"""
import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--roller_damping', type=float, default=1e-4)
parser.add_argument('--roller_friction', type=float, default=0.06, help='dry (Coulomb) friction torque of the roller joints [Nm], always applied (0 = none; 0.06 = asset value)')
parser.add_argument('--track_width', type=float, default=0.395)
parser.add_argument('--csv', default=None, help='log t, command and measured velocities to this CSV')
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
R = 0.015
DEMO = [(1.0, 0.0, 0.0), (4.0, 0.3, 0.0), (4.0, 0.0, 0.8), (4.0, 0.3, 0.5), (4.0, -0.3, 0.0), (4.0, 0.0, -0.8),
        (2.0, 0.0, 0.0)]
T_END = sum(d for d, _, _ in DEMO)
STANCE = {'LF_HFE': 1.47, 'LH_HFE': -1.47, 'RF_HFE': -1.47, 'RH_HFE': 1.47,
          'body_left_F_ankle': 0.10, 'body_right_F_ankle': 0.10,
          'body_left_H_ankle': -0.10, 'body_right_H_ankle': -0.10}
CAM_OFFSET = np.array([1.1, 1.3, 0.75])


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mat = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(1.0)
    m.CreateDynamicFrictionAttr().Set(1.0)
    UsdGeom.Xform.Define(stage, '/World/Ground')
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')
    plane.CreateAxisAttr().Set('Z')
    plane.CreateWidthAttr().Set(40.0)
    plane.CreateLengthAttr().Set(40.0)
    plane.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.57, 0.6)])
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    # 1 m grid lines on the floor, so motion and rotation are visible
    for i in range(-10, 11):
        for name, size, pos in ((f'gx{i + 10}', (0.01, 20.0, 0.002), (i, 0, 0.001)),
                                (f'gy{i + 10}', (20.0, 0.01, 0.002), (0, i, 0.001))):
            c = UsdGeom.Cube.Define(stage, f'/World/Grid/{name}')
            c.CreateSizeAttr().Set(1.0)
            c.CreateDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.37, 0.4)])
            xf = UsdGeom.Xformable(c.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr().Set(900.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr().Set(2500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
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
    for p in stage.Traverse():
        if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr().Set(1.0)
    SimulationManager.setup_simulation(dt=DT, device='cpu')
    await app_utils.update_app_async()
    return Articulation(str(roots[0].GetPath()))


robot = app.run_coroutine(build())
cam = RtxCamera('/World/FollowCam', tick_rate=0.0)
cam_usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath('/World/FollowCam'))
cam_usd.GetFocalLengthAttr().Set(18.0)             # mm, with a 20.955 mm aperture: ~60 deg horizontal FOV
cam_usd.GetHorizontalApertureAttr().Set(20.955)
cam_usd.GetVerticalApertureAttr().Set(20.955 * 720 / 1280)
cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
print('camera focal', cam_usd.GetFocalLengthAttr().Get(), flush=True)
sensor = CameraSensor(cam, resolution=(720, 1280), annotators=['rgb'])
app_utils.play(commit=True)
app.update()

names = robot.dof_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
hfe = [idx[n] for n in ('LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE')]
ankles = [i for n, i in idx.items() if n.endswith('_ankle')]
# mimic followers (cylinder model) have no drive: only the leader wheel of each track is commanded
FOLLOWERS = set()
for p in omni.usd.get_context().get_stage().Traverse():
    rel = p.GetRelationship('newton:mimicJoint') if p.GetName().startswith('joint_wheel_') else None
    if rel and rel.GetTargets():
        FOLLOWERS.add(p.GetName())
left = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'LEFT' in n and n not in FOLLOWERS]
right = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'RIGHT' in n and n not in FOLLOWERS]
rollers = [i for n, i in idx.items() if '_roller_' in n]
kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
kp[hfe] = 100.0; kd[hfe] = 10.0; fmax[hfe] = 5.0
kd[ankles] = 0.05   # passive ankles, as in Gazebo and in the asset
per_track = len(left) / 2                          # driven wheels per track: 4 (rollers) or 1 (cylinder)
kd[left + right] = 2.0 / per_track; fmax[left + right] = 40.0 / per_track
kd[rollers] = args.roller_damping
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
if rollers:
    tau = np.full((1, len(rollers)), args.roller_friction, np.float32)
    robot.set_dof_friction_properties(static_frictions=tau, dynamic_frictions=tau, dof_indices=rollers)
robot.set_dof_max_efforts(fmax[None])
arm = np.zeros(N, np.float32); arm[left + right] = 0.001
robot.set_dof_armatures(arm[None])
q0 = np.zeros(N, np.float32)
for n, v in STANCE.items():
    q0[idx[n]] = v
robot.set_dof_positions(q0[None])
robot.set_dof_position_targets(q0[None])


def command(t):
    for dur, v, w in DEMO:
        if t < dur:
            return v, w
        t -= dur
    return 0.0, 0.0


def look_at(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.array([x, y, z]).T
    qw = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [qw, (m[2, 1] - m[1, 2]) / (4 * qw), (m[0, 2] - m[2, 0]) / (4 * qw), (m[1, 0] - m[0, 1]) / (4 * qw)]


try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
except OSError:
    font = ImageFont.load_default()

t0 = SimulationManager.get_simulation_time()
n_saved = 0
log = []
MODEL = f'roller wheels, roller friction {args.roller_friction:g} Nm' if rollers else 'cylinder wheels (mimic)'
cmd = (0.0, 0.0)
while True:
    t = SimulationManager.get_simulation_time() - t0
    if t > T_END:
        break
    v, w = command(t)
    if (v, w) != cmd:
        cmd = (v, w)
        vl, vr = v - w * args.track_width / 2, v + w * args.track_width / 2
        tgt = np.zeros(N, np.float32); tgt[left] = -vl / R; tgt[right] = vr / R
        robot.set_dof_velocity_targets(tgt[None])
    p, q = robot.get_world_poses()
    p = p.numpy()[0]
    target = np.array([p[0], p[1], 0.12])
    cam.set_world_poses(positions=[(target + CAM_OFFSET).tolist()], orientations=[look_at(target + CAM_OFFSET, target)])
    app.update()
    data, _ = sensor.get_data('rgb')
    if data is None:
        continue
    lin, ang = robot.get_velocities()
    qw, qx, qy, qz = q.numpy()[0]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    lin = lin.numpy()[0]
    vx = math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1]
    wz = float(ang.numpy()[0][2])
    img = Image.fromarray(np.asarray(data.numpy())[..., :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 84], fill=(255, 255, 255))
    d.text((20, 10), f'Isaac Sim (PhysX) - {MODEL} - t = {t:5.2f} s', fill=(20, 20, 20), font=font)
    d.text((20, 46), f'command  v = {v:+.2f} m/s  w = {w:+.2f} rad/s      measured  v = {vx:+.2f} m/s  w = {wz:+.2f} rad/s',
           fill=(20, 20, 20), font=font)
    log.append([round(t, 3), v, w, round(vx, 4), round(wz, 4)])
    img.save(os.path.join(args.out, f'{t:08.3f}.png'))
    n_saved += 1
if args.csv:
    import csv
    with open(args.csv, 'w', newline='') as fh:
        csv.writer(fh).writerows([['t', 'v_cmd', 'w_cmd', 'vx', 'wz']] + log)
print('frames saved', n_saved, 'sim time', round(SimulationManager.get_simulation_time() - t0, 2), flush=True)
app.close()
