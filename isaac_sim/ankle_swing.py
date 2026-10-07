"""Traquad suspended with the base fixed and the hips locked, to watch the tracks on their ankles: released from a
tilt (pendulum test of the real robot) and/or driven by a sequence of track speed commands.

The base is fixed in the air (articulation root moved to a fixed joint to the world), the HFE are held by stiff
drives at the stance. Ankle = track motor coupling as in the asset and in Isaac Lab: velocity drive towards the
sprocket speed v / SPROCKET_RADIUS when the track is in the air (ankle_control.py: velocity controller, gain 0.01 N m s/rad,
max 0.05 N m), plus the passive damping of the pivot --ankle_damping (0.01 N m s/rad). With no command the controller
only damps: the hanging track swings back to its hanging pose. All four ankles
start --offset rad away from the flat pose; --sequence "dur:v,dur:v,..." gives the track speed v [m/s] (same on the
four tracks, forward positive) for each duration [s]. Frames from a side camera on the front-left track are saved
as PNG (named by simulated time) with command and ankle angle on them; --csv logs the four ankle angles.

usage: ./isaac.sh ankle_swing.py --usd <robot.usda> --out <frames_dir> [--csv log.csv] [--offset 0.25]
                                 [--sequence "1:0,3:0.05,3:0.1"] [--duration 4]
"""
import argparse
from ankle_control import PASSIVE_DAMPING, SPROCKET_RADIUS, AnkleController
import csv
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--out', default=None, help='frames directory (no frames if omitted)')
parser.add_argument('--csv', default=None)
parser.add_argument('--offset', type=float, default=0.25, help='initial ankle tilt from the flat pose [rad]')
parser.add_argument('--ankle_damping', type=float, default=None, help='passive damping of the pivot [N m s/rad] (default: ankle_control)')
parser.add_argument('--no_control', action='store_true', help='passive ankles only (no velocity control in the air)')
parser.add_argument('--sequence', default='', help='track speed commands "duration:v,..." [s:m/s]; default: none')
parser.add_argument('--duration', type=float, default=4.0, help='[s]')
parser.add_argument('--hfe', type=float, default=1.47, help='HFE stance [rad]; the flat pose of the ankle is pi/2 - hfe')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'width': 1280, 'height': 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

SimulationManager.switch_physics_engine('physx')
DT = 1.0 / 200.0
Z0 = 0.6                      # base height [m]
HFE = {'LF_HFE': args.hfe, 'LH_HFE': -args.hfe, 'RF_HFE': -args.hfe, 'RH_HFE': args.hfe}
R_WHEEL = 0.015
SEQ = [(float(d), float(v)) for d, v in (s.split(':') for s in args.sequence.split(',') if s)]
if SEQ:
    args.duration = sum(d for d, _ in SEQ)
_flat = math.pi / 2 - args.hfe   # track flat (horizontal) when |HFE| + |ankle| = pi/2
FLAT = {'body_left_F_ankle': _flat, 'body_right_F_ankle': _flat,
        'body_left_H_ankle': -_flat, 'body_right_H_ankle': -_flat}


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    plane = UsdGeom.Plane.Define(stage, '/World/Ground/Plane')    # visual reference only, far below the tracks
    plane.CreateAxisAttr().Set('Z')
    plane.CreateWidthAttr().Set(20.0)
    plane.CreateLengthAttr().Set(20.0)
    plane.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.57, 0.6)])
    UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr().Set(900.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr().Set(2500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
    stage_utils.add_reference_to_stage(usd_path=args.usd, path='/World/robot')
    robot_prim = stage.GetPrimAtPath('/World/robot')
    robot_prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    UsdGeom.XformCommonAPI(robot_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, Z0))
    await app_utils.update_app_async()
    # fixed base: the articulation root goes from the base link to a fixed joint between the world and the base
    base = next(p for p in stage.Traverse()
                if str(p.GetPath()).startswith('/World/robot') and p.HasAPI(UsdPhysics.ArticulationRootAPI))
    base.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    fix = UsdPhysics.FixedJoint.Define(stage, '/World/BaseFix')
    fix.CreateBody1Rel().SetTargets([base.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(fix.GetPrim())
    px = PhysxSchema.PhysxArticulationAPI.Apply(fix.GetPrim())
    px.CreateEnabledSelfCollisionsAttr().Set(False)
    px.CreateSolverPositionIterationCountAttr().Set(16)
    px.CreateSolverVelocityIterationCountAttr().Set(4)
    SimulationManager.setup_simulation(dt=DT, device='cpu')
    await app_utils.update_app_async()
    return Articulation('/World/BaseFix')


robot = app.run_coroutine(build())
sensor = None
if args.out:
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(args.out, exist_ok=True)
    cam = RtxCamera('/World/SideCam', tick_rate=0.0)
    cam_usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath('/World/SideCam'))
    cam_usd.GetFocalLengthAttr().Set(24.0)
    cam_usd.GetHorizontalApertureAttr().Set(20.955)
    cam_usd.GetVerticalApertureAttr().Set(20.955 * 720 / 1280)
    cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    sensor = CameraSensor(cam, resolution=(720, 1280), annotators=['rgb'])
app_utils.play(commit=True)
app.update()

names = robot.dof_names
N = len(names)
idx = {n: i for i, n in enumerate(names)}
hfe = [idx[n] for n in HFE]
ankles = [idx[n] for n in FLAT]
wheels = [i for n, i in idx.items() if n.startswith('joint_wheel_')]
kp = np.zeros(N, np.float32); kd = np.zeros(N, np.float32); fmax = np.full(N, 1e3, np.float32)
kp[hfe] = 1e4; kd[hfe] = 100.0                      # hips locked
kd[ankles] = PASSIVE_DAMPING if args.ankle_damping is None else args.ankle_damping   # passive ankles
ankle_ctrl = AnkleController(robot, enabled=not args.no_control)
kd[wheels] = 0.5; fmax[wheels] = 0.375              # track motor: 1.5 N m per track
robot.set_dof_gains(stiffnesses=kp[None], dampings=kd[None])
robot.set_dof_max_efforts(fmax[None])
q0 = np.zeros(N, np.float32)
for n, v in HFE.items():
    q0[idx[n]] = v
for n, v in FLAT.items():
    q0[idx[n]] = v + args.offset                    # same tilt direction for the four tracks
robot.set_dof_positions(q0[None])
robot.set_dof_position_targets(q0[None])
robot.set_dof_velocities(np.zeros((1, N), np.float32))
robot.set_dof_velocity_targets(np.zeros((1, N), np.float32))
lo, hi = (x.numpy()[0] for x in robot.get_dof_limits())
print('ankle limits', {n: (round(float(lo[idx[n]]), 3), round(float(hi[idx[n]]), 3)) for n in FLAT}, flush=True)


def look_at(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.array([x, y, z]).T
    qw = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [qw, (m[2, 1] - m[1, 2]) / (4 * qw), (m[0, 2] - m[2, 0]) / (4 * qw), (m[1, 0] - m[0, 1]) / (4 * qw)]


if sensor is not None:
    tf = robot._physics_articulation_view.get_link_transforms().numpy()[0]
    target = tf[robot.link_names.index('body_left_F'), :3].astype(float)
    eye = target + np.array([0.05, 0.75, 0.05])
    cam.set_world_poses(positions=[eye.tolist()], orientations=[look_at(eye, target)])
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    except OSError:
        font = ImageFont.load_default()

left = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'LEFT' in n]
right = [i for n, i in idx.items() if n.startswith('joint_wheel_') and 'RIGHT' in n]
view = robot._physics_articulation_view
LN = robot.link_names
frame_link, wheel_link = LN.index('body_left_F'), LN.index('wheel_2_LEFT_F')


def command(t):
    for d, v in SEQ:
        if t < d:
            return v
        t -= d
    return 0.0


def send(v):
    vel = np.zeros(N, np.float32)
    vel[left] = -v / R_WHEEL                      # wheel axes: -y on the left, +y on the right
    vel[right] = v / R_WHEEL
    robot.set_dof_velocity_targets(vel[None])
    ankle_ctrl.set_speeds(v, v)


t0 = SimulationManager.get_simulation_time()
log = []
cmd = None
while True:
    t = SimulationManager.get_simulation_time() - t0
    if t > args.duration:
        break
    v = command(t)
    if v != cmd:
        send(v)
        cmd = v
    if sensor is not None:
        app.update()
        data, _ = sensor.get_data('rgb')
    else:
        SimulationManager.step(steps=1)
        data = None
    q = robot.get_dof_positions().numpy()[0]
    rel = [float(q[i] - FLAT[n]) for n, i in zip(FLAT, ankles)]
    lv = view.get_link_velocities().numpy()[0]
    log.append([round(t, 4), v] + [round(r, 5) for r in rel] + [round(float(lv[frame_link, 4]), 4), round(float(lv[wheel_link, 4]), 4)])
    if data is None:
        continue
    img = Image.fromarray(np.asarray(data.numpy())[..., :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 84], fill=(255, 255, 255))
    d.text((20, 10), f'Suspended Traquad, hips locked at HFE {args.hfe:.2f} rad - track motor coupled to the ankle - t = {t:5.2f} s',
           fill=(20, 20, 20), font=font)
    d.text((20, 46), f'track speed command {v:+.2f} m/s      front-left ankle {math.degrees(rel[0]):+6.1f} deg from flat',
           fill=(20, 20, 20), font=font)
    img.save(os.path.join(args.out, f'{t:08.3f}.png'))

if args.csv:
    with open(args.csv, 'w', newline='') as f:
        csv.writer(f).writerows([['t', 'v_cmd'] + [n + '_from_flat' for n in FLAT] + ['frame_wy', 'wheel_wy']] + log)
a = np.array(log)
fl = a[:, 2]
for vv in sorted(set(a[:, 1])):
    m = (a[:, 1] == vv)
    print(f'RESULT v_cmd {vv:+.2f}: end ankle {math.degrees(a[m, 2][-1]):+.1f} deg | mean world w_y frame '
          f'{a[m, 6].mean():+.3f} wheel {a[m, 7].mean():+.2f} rad/s', flush=True)
print(f'RESULT front-left ankle from flat: start {math.degrees(fl[0]):+.1f} deg, end {math.degrees(fl[-1]):+.1f} deg, '
      f'min {math.degrees(fl.min()):+.1f} max {math.degrees(fl.max()):+.1f} deg, max speed '
      f'{np.abs(np.diff(fl) / np.diff(a[:, 0])).max():.2f} rad/s', flush=True)
app.close()
