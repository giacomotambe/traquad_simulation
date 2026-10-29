"""Render a video of a replay (real_replay.py --frames) next to the real robot.

Both robots are posed kinematically on every frame (no gravity, no collisions): the simulated one from the frame log
of the replay (grey), the real one (orange) from the measured data (real_id/ghost.py: Qualisys base pose, hip encoders, track
speeds), aligned so that both start from the same pose. Ankles of the real robot (no encoder) are drawn flat. Frames
are PNG (1280x720, 30 per second) with the joystick command, R2 state and both velocities written on them.

usage: ./isaac.sh render_replay.py --frames frames.npz --ghost ghost.npz --out <dir> [--title "..."] [--side 0]
       ffmpeg -framerate 30 -i <dir>/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 23 video.mp4
"""
import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--frames', required=True)
parser.add_argument('--ghost', default=None)
parser.add_argument('--out', required=True)
parser.add_argument('--title', default='')
parser.add_argument('--usd', default=None)
parser.add_argument('--hfe_offset', type=float, default=-0.05, help='geometric - encoder hip angle (in the asset)')
parser.add_argument('--side', type=float, default=0.0,
                    help='draw the real robot this far to the left of where it is [m] (0: on top of each other)')
parser.add_argument('--cmd_label', default='joystick', help='name of the command source (not joystick: no R2 label)')
parser.add_argument('--cam', type=float, nargs=3, default=[1.5, 1.8, 1.15], help='camera offset from the target [m]')
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
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

SimulationManager.switch_physics_engine('physx')
HERE = os.path.dirname(os.path.abspath(__file__))
USD = args.usd or os.path.join(HERE, 'assets/traquad/traquad.usda')
F = dict(np.load(args.frames))
G = dict(np.load(args.ghost)) if args.ghost else None
TRACKS = ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')
SIGMA = [1, -1, -1, 1]
os.makedirs(args.out, exist_ok=True)


def kinematic(stage, root):
    """No gravity, no collisions: the robot is only posed."""
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr().Set(True)


async def build():
    await stage_utils.create_new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    plane = UsdGeom.Mesh.Define(stage, '/World/Floor')
    plane.CreatePointsAttr([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.52, 0.54)])
    for i in range(-30, 31):   # 0.5 m grid
        for name, size, pos in ((f'gx{i + 30}', (0.008, 30.0, 0.002), (i * 0.5, 0, 0.001)),
                                (f'gy{i + 30}', (30.0, 0.008, 0.002), (0, i * 0.5, 0.001))):
            c = UsdGeom.Cube.Define(stage, f'/World/Grid/{name}')
            c.CreateSizeAttr().Set(1.0)
            c.CreateDisplayColorAttr().Set([Gf.Vec3f(0.44, 0.46, 0.49)])
            xf = UsdGeom.Xformable(c.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr().Set(450.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr().Set(1300.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
    roots = {}
    for name in (['sim', 'ghost'] if G is not None else ['sim']):
        path = f'/World/{name}'
        stage_utils.add_reference_to_stage(usd_path=USD, path=path)
        prim = stage.GetPrimAtPath(path)
        prim.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0.0, 5.0 if name == 'ghost' else 0.0, 0.5))
    await app_utils.update_app_async()
    # one flat colour per robot (un-instance the meshes to bind a material): simulation grey, real robot orange
    for name, color in (('sim', (0.36, 0.38, 0.42)), ('ghost', (0.85, 0.28, 0.03))):
        if name == 'ghost' and G is None:
            continue
        g = stage.GetPrimAtPath(f'/World/{name}')
        changed = True
        while changed:
            changed = False
            for p in Usd.PrimRange(g):
                if p.IsInstanceable():
                    p.SetInstanceable(False)
                    changed = True
        mat = UsdShade.Material.Define(stage, f'/World/Looks/{name}')
        sh = UsdShade.Shader.Define(stage, f'/World/Looks/{name}/shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.45)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        for p in Usd.PrimRange(g):
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    await app_utils.update_app_async()
    for name in (['sim', 'ghost'] if G is not None else ['sim']):
        kinematic(stage, f'/World/{name}')
        rr = [str(p.GetPath()) for p in stage.Traverse()
              if str(p.GetPath()).startswith(f'/World/{name}/') and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(rr[0])).CreateEnabledSelfCollisionsAttr().Set(False)
        roots[name] = rr[0]
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device='cpu')
    await app_utils.update_app_async()
    return {k: Articulation(v) for k, v in roots.items()}


arts = app.run_coroutine(build())
cam = RtxCamera('/World/VideoCam', tick_rate=0.0)
cam_usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath('/World/VideoCam'))
cam_usd.GetFocalLengthAttr().Set(18.0)
cam_usd.GetHorizontalApertureAttr().Set(20.955)
cam_usd.GetVerticalApertureAttr().Set(20.955 * 720 / 1280)
cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
sensor = CameraSensor(cam, resolution=(720, 1280), annotators=['rgb'])
app_utils.play(commit=True)
app.update()
sim = arts['sim']
names = list(sim.dof_names)
N = len(names)
if list(F['dof_names']) != names:   # frames logged in another joint order (Isaac Lab): reorder by name
    assert sorted(F['dof_names']) == sorted(names)
    F['q'] = F['q'][:, [list(F['dof_names']).index(n) for n in names]]
idx = {n: i for i, n in enumerate(names)}
for a in arts.values():
    a.set_dof_gains(stiffnesses=np.zeros((1, N), np.float32), dampings=np.zeros((1, N), np.float32))
hfe = [idx[f'{l}_HFE'] for l in ('LF', 'LH', 'RF', 'RH')]
ankles = [idx[f"body_{t.split('_')[0].lower()}_{t.split('_')[1]}_ankle"] for t in TRACKS]
wheels = [[idx[f'joint_wheel_{k}_{t}'] for k in (1, 2, 3, 4)] for t in TRACKS]

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    font_b = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 25)
except OSError:
    font = font_b = ImageFont.load_default()


def look_at(eye, target):
    f = target - eye
    f = f / np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.array([x, y, z]).T
    qw = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [qw, (m[2, 1] - m[1, 2]) / (4 * qw), (m[0, 2] - m[2, 0]) / (4 * qw), (m[1, 0] - m[0, 1]) / (4 * qw)]


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gi(key, t):
    v = G[key]
    return np.array([np.interp(t, G['t'], v[:, j]) for j in range(v.shape[1])]) if v.ndim == 2 \
        else float(np.interp(t, G['t'], v))


t_frames = F['t']
yaw0 = yaw_of(F['quat'][0])
if G is not None:
    dyaw = yaw0 - gi('yaw', t_frames[0])
    xr0, yr0 = gi('x', t_frames[0]), gi('y', t_frames[0])
    side = np.array([-math.sin(yaw0), math.cos(yaw0)]) * args.side
z0 = float(F['pos'][0, 2])
n_saved = 0
for i, t in enumerate(t_frames):
    p_s, q_s = F['pos'][i], F['quat'][i]
    sim.set_world_poses(positions=[p_s.tolist()], orientations=[q_s.tolist()])
    sim.set_dof_positions(F['q'][i][None].astype(np.float32))
    target = np.array([p_s[0], p_s[1], 0.12])
    vx_r = wz_r = None
    if G is not None:
        c, s = math.cos(dyaw), math.sin(dyaw)
        dx, dy = gi('x', t) - xr0, gi('y', t) - yr0
        gx = F['pos'][0, 0] + c * dx - s * dy + side[0]
        gy = F['pos'][0, 1] + s * dx + c * dy + side[1]
        th = gi('yaw', t) + dyaw
        qg = np.zeros(N, np.float32)
        qh = gi('q_hfe', t)
        qg[hfe] = qh
        for j, tr in enumerate(TRACKS):
            alpha = SIGMA[j] * qh[j] + args.hfe_offset
            qg[ankles[j]] = (math.pi / 2 - alpha) * (1 if tr.endswith('_F') else -1)
            qg[wheels[j]] = (gi('wheel_angle', t)[j] + math.pi) % (2 * math.pi) - math.pi
        arts['ghost'].set_world_poses(positions=[[gx, gy, z0]], orientations=[[math.cos(th / 2), 0, 0, math.sin(th / 2)]])
        arts['ghost'].set_dof_positions(qg[None])
        vx_r, wz_r = gi('vx', t), gi('wz', t)
        target = np.array([(p_s[0] + gx) / 2, (p_s[1] + gy) / 2, 0.12])
    eye = target + np.array(args.cam)
    cam.set_world_poses(positions=[eye.tolist()], orientations=[look_at(eye, target)])
    data = None
    for _ in range(4):
        for a in arts.values():
            a.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
            a.set_dof_velocities(np.zeros((1, N), np.float32))
        app.update()
        data, _ = sensor.get_data('rgb')
        if data is not None:
            break
    if data is None:
        continue
    img = Image.fromarray(np.asarray(data.numpy())[..., :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 116], fill=(255, 255, 255))
    d.text((20, 8), f'{args.title}    t = {t - t_frames[0]:5.1f} s', fill=(20, 20, 20), font=font_b)
    if args.cmd_label == 'joystick':
        r2 = bool(F['r2'][i])
        d.text((1130, 8), 'R2 ON' if r2 else 'R2 off', fill=(0, 140, 90) if r2 else (130, 130, 130), font=font_b)
    v_c, w_c = F['cmd'][i]
    d.text((20, 46), f'{args.cmd_label}   v {v_c:+.2f} m/s   w {w_c:+.2f} rad/s', fill=(20, 20, 20), font=font)
    d.text((20, 80), f'simulation (grey)   v {F["vx"][i]:+.2f}   w {F["wz"][i]:+.2f}', fill=(80, 80, 80), font=font)
    if vx_r is not None:
        d.text((600, 80), f'real robot (orange, Qualisys)   v {vx_r:+.2f}   w {wz_r:+.2f}', fill=(215, 90, 20), font=font)
    img.save(os.path.join(args.out, f'{n_saved:05d}.png'))
    n_saved += 1
print('frames saved', n_saved, 'of', len(t_frames), flush=True)
app.close()
