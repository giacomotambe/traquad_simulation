"""Evaluate a TraQuad policy of the mixed task: one robot per terrain type and level, plus command tracking on flat.

Terrain test: each robot starts at the centre of its tile facing +x and is commanded straight ahead at --speed for
--duration seconds. Past 3.5 m it leaves its tile (the next level starts there), so the test measures the time to
reach 3 m from the start: the obstacles of the tile start 1-1.5 m from the centre. The centre of the pyramids is the top (stairs, slope) or the bottom (inverted ones): driving out
goes down or up. Printed per robot: distance travelled along +x, mean forward speed, fall (base contact), largest
roll/pitch and HFE deviation from the default stance.
Command test (flat tile): forward, backward, turns and rotation in place, mean measured v and w against the command.
Videos (--clips): the camera follows the chosen robots, one clip each.

Run from isaaclab_traquad/:
    OMNI_KIT_ACCEPT_EULA=YES python ../isaaclab_overlay/scripts/eval_traquad.py --checkpoint logs/rsl_rl/traquad_mixed/<run>/model_X.pt
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Mixed-TraQuad")
parser.add_argument("--speed", type=float, default=0.5, help="forward command of the terrain test [m/s]")
parser.add_argument("--duration", type=float, default=10.0, help="[s] per test")
parser.add_argument("--static", type=float, default=1.0, help="static friction of all robots")
parser.add_argument("--dynamic", type=float, default=0.7, help="dynamic friction of all robots")
parser.add_argument("--levels", default="0,3,6,9")
parser.add_argument("--repeats", type=int, default=4, help="robots per case (side by side)")
parser.add_argument("--clips", default="", help="clips 'type:level,...' (types: flat rough stairs stairs_inv boxes slope slope_inv)")
parser.add_argument("--out", default="eval", help="output folder (videos)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.clips:
    args.enable_cameras = True
    args.visualizer = "kit"   # without a visualizer the robot poses are not synced to the renderer
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# column of each terrain type in the 20 columns of the generator (curriculum: types by cumulative proportion)
GOAL = 3.0   # [m] from the centre of the tile
TYPES = {"flat": 1, "rough": 4, "stairs": 7, "stairs_inv": 10, "boxes": 12, "slope": 15, "slope_inv": 18}
COMMANDS = [("forward 1.0", 1.0, 0.0), ("forward 0.5", 0.5, 0.0), ("backward 0.5", -0.5, 0.0),
            ("turn 0.5 / 0.5", 0.5, 0.5), ("turn 0.3 / -0.8", 0.3, -0.8), ("in place 1.0", 0.0, 1.0),
            ("in place -0.5", 0.0, -0.5), ("stand", 0.0, 0.0)]
levels = [int(x) for x in args.levels.split(",")]
terrain_cases = [(t, lv) for t in TYPES for lv in levels]
n_terrain = len(terrain_cases)
n_base = n_terrain + len(COMMANDS)
R = args.repeats
num_envs = n_base * R   # robot of case b, repeat r: index r * n_base + b

env_cfg = parse_env_cfg(args.task, num_envs=num_envs)
env_cfg.events.ground_friction.params = {"static_friction_range": (args.static, args.static),
                                         "dynamic_friction_range": (args.dynamic, args.dynamic), "num_buckets": 1}
env_cfg.events.push_robot = None
env_cfg.events.base_external_force_torque = None
env_cfg.curriculum.terrain_levels = None
env_cfg.scene.terrain.terrain_generator.curriculum = True   # keep the ordered grid (rows = levels, cols = types)
env_cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
env_cfg.commands.base_velocity.debug_vis = False
env_cfg.episode_length_s = 1e4
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
env = gym.make(args.task, cfg=env_cfg)
u = env.unwrapped
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = create_rsl_rl_runner(wrapped, agent_cfg)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=u.device)

robot, terrain = u.scene["robot"], u.scene.terrain
CAM_OFFSET = np.array([-1.0, -1.5, 0.8])   # follow camera, from behind on the left
if args.clips:
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.sensors.experimental.rtx", True
    )
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
    from pxr import Gf, UsdGeom

    import omni.usd

    cam = RtxCamera("/World/FollowCam", tick_rate=0.0)
    cam_usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/World/FollowCam"))
    cam_usd.GetFocalLengthAttr().Set(18.0)
    cam_usd.GetHorizontalApertureAttr().Set(20.955)
    cam_usd.GetVerticalApertureAttr().Set(20.955 * 540 / 960)
    cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    cam_sensor = CameraSensor(cam, resolution=(540, 960), annotators=["rgb"])


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.array([x, y, z]).T
    qw = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [qw, (m[2, 1] - m[1, 2]) / (4 * qw), (m[0, 2] - m[2, 0]) / (4 * qw), (m[1, 0] - m[0, 1]) / (4 * qw)]


def overlay(img, line1, line2):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 52], fill=(255, 255, 255))
    d.text((10, 4), line1, fill=(20, 20, 20), font=font)
    d.text((10, 28), line2, fill=(20, 20, 20), font=font)
    return np.asarray(im)


def clip_label(t, lvl):
    if t == "commands":
        return f"policy {os.path.basename(args.checkpoint)} - flat, command '{COMMANDS[int(lvl)][0]}'"
    return (f"policy {os.path.basename(args.checkpoint)} - {t} level {lvl} - friction static {args.static} "
            f"dynamic {args.dynamic}")


def grab(idx):
    target = robot.data.root_pos_w.torch[idx].cpu().numpy().astype(float)
    eye = target + CAM_OFFSET
    cam.set_world_poses(positions=[eye.tolist()], orientations=[look_at(eye, target)])
    app.update()
    data, _ = cam_sensor.get_data("rgb")
    return None if data is None else np.asarray(data.numpy())[..., :3].astype(np.uint8)

dev = u.device
hfe = robot.find_joints(".*HFE")[0]
cmd_term = u.command_manager.get_term("base_velocity")
lv = torch.tensor([lvl for _, lvl in terrain_cases] + [0] * len(COMMANDS), device=dev).repeat(R)
ty = torch.tensor([TYPES[t] for t, _ in terrain_cases] + [TYPES["flat"]] * len(COMMANDS), device=dev).repeat(R)
origins = terrain.terrain_origins[lv, ty].clone()
rep_id = torch.arange(num_envs, device=dev) // n_base
case_id = torch.arange(num_envs, device=dev) % n_base
is_cmd = case_id >= n_terrain
# repeats side by side across the direction of motion; command robots spread over the flat tiles of levels 0-1
origins[:, 1] += (rep_id.float() - (R - 1) / 2) * 0.8
origins[is_cmd, 1] = terrain.terrain_origins[0, TYPES["flat"], 1] + (case_id[is_cmd] - n_terrain - 3.5).float() * 0.9
origins[is_cmd, 0] = terrain.terrain_origins[0, TYPES["flat"], 0] + (rep_id[is_cmd].float() - (R - 1) / 2) * 2.0
terrain.env_origins[:] = origins   # automatic resets (after a fall) also go back there
command = torch.zeros(num_envs, 3, device=dev)
command[~is_cmd, 0] = args.speed
for j, (_, v, w) in enumerate(COMMANDS):
    sel = case_id == n_terrain + j
    command[sel, 0], command[sel, 2] = v, w


@torch.inference_mode()
def restart():
    """Put every robot back at its start, default joints, zero speed; reset the policy memory."""
    ids = torch.arange(num_envs, device=dev)
    pose = torch.zeros(num_envs, 7, device=dev)
    pose[:, :3] = origins + torch.tensor([0.0, 0.0, 0.29], device=dev)
    pose[:, 6] = 1.0   # quaternion (x, y, z, w)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_envs, 6, device=dev), env_ids=ids)
    robot.write_joint_state_to_sim_index(position=robot.data.default_joint_pos.torch.clone(),
                                         velocity=torch.zeros_like(robot.data.default_joint_vel.torch), env_ids=ids)
    for term in u.action_manager._terms.values():
        term.reset(ids)
    policy.reset(torch.ones(num_envs, dtype=torch.bool, device=dev))
    cmd_term.vel_command_b[:] = command
    return wrapped.get_observations()


@torch.inference_mode()
def run(follow=None, frames=None, label=""):
    obs = restart()
    for _ in range(3):   # let the written state settle into the observations
        cmd_term.vel_command_b[:] = command
        obs, _, _, _ = wrapped.step(torch.zeros_like(policy(obs)))
    policy.reset(torch.ones(num_envs, dtype=torch.bool, device=dev))
    start = robot.data.root_pos_w.torch.clone()
    fell = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    max_tilt = torch.zeros(num_envs, device=dev)
    max_hfe = torch.zeros(num_envs, device=dev)
    vx, wz = [], []
    t_reach = torch.full((num_envs,), float("nan"), device=dev)
    steps = int(args.duration / u.step_dt)
    sensor = u.scene.sensors["contact_forces"]
    base_id = sensor.find_sensors("base_link")[0]
    for k in range(steps):
        cmd_term.vel_command_b[:] = command
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        force = torch.linalg.norm(sensor.data.net_forces_w.torch[:, base_id], dim=-1).max(dim=1).values
        fell |= force > 0.1
        g = robot.data.projected_gravity_b.torch
        max_tilt = torch.maximum(max_tilt, torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))))
        dq = (robot.data.joint_pos.torch[:, hfe] - robot.data.default_joint_pos.torch[:, hfe]).abs().max(dim=1).values
        max_hfe = torch.maximum(max_hfe, torch.rad2deg(dq))
        reached = (robot.data.root_pos_w.torch[:, 0] - start[:, 0] >= GOAL) & torch.isnan(t_reach)
        t_reach[reached] = (k + 1) * u.step_dt
        if k * u.step_dt >= 1.0:
            vx.append(robot.data.root_lin_vel_b.torch[:, 0].clone())
            wz.append(robot.data.root_ang_vel_b.torch[:, 2].clone())
        if frames is not None:
            img = grab(follow)
            if img is not None:
                v_b = robot.data.root_lin_vel_b.torch[follow]
                w_b = robot.data.root_ang_vel_b.torch[follow, 2]
                frames.append(overlay(img, f"{label}   t = {(k + 1) * u.step_dt:5.2f} s",
                                      f"command v = {command[follow, 0]:+.2f} m/s  w = {command[follow, 2]:+.2f} rad/s"
                                      f"     measured v = {v_b[0]:+.2f} m/s  w = {w_b:+.2f} rad/s"))
    disp = robot.data.root_pos_w.torch - start
    return disp, fell, max_tilt, max_hfe, torch.stack(vx).mean(0), torch.stack(wz).mean(0), t_reach


disp, fell, tilt, dhfe, vx, wz, t_reach = run()
print(f"\nEVAL {args.checkpoint} | friction static {args.static} dynamic {args.dynamic} | command {args.speed} m/s, "
      f"{args.duration} s")
print(f"repeats {R} per case; reached = got 3 m from the start within {args.duration} s")
print("TERRAIN type        level  reached  t_3m_median[s]  dist_x_mean[m]  falls  max_tilt[deg]  max_dHFE[deg]")
for b, (t, lvl) in enumerate(terrain_cases):
    ids = torch.nonzero(case_id == b).squeeze(1)
    tr = t_reach[ids]
    ok = ~torch.isnan(tr)
    med = f"{tr[ok].median().item():14.1f}" if ok.any() else "             -"
    print(f"TERRAIN {t:11s} {lvl:5d} {int(ok.sum())}/{R:<6d} {med} {disp[ids, 0].mean().item():15.2f} "
          f"{int(fell[ids].sum()):6d} {tilt[ids].max().item():14.1f} {dhfe[ids].max().item():14.1f}")
print("COMMAND name              v_cmd  w_cmd   v_meas (min-max)     w_meas (min-max)    falls")
for j, (name, v, w) in enumerate(COMMANDS):
    ids = torch.nonzero(case_id == n_terrain + j).squeeze(1)
    print(f"COMMAND {name:16s} {v:6.2f} {w:6.2f}  {vx[ids].mean().item():6.2f} ({vx[ids].min().item():5.2f} {vx[ids].max().item():5.2f})"
          f"  {wz[ids].mean().item():6.2f} ({wz[ids].min().item():5.2f} {wz[ids].max().item():5.2f})  {int(fell[ids].sum()):4d}")

if args.clips:
    import subprocess
    os.makedirs(args.out, exist_ok=True)
    for clip in args.clips.split(","):
        t, lvl = clip.split(":")
        idx = terrain_cases.index((t, int(lvl))) if t != "commands" else n_terrain + int(lvl)   # first repeat
        frames = []
        run(follow=idx, frames=frames, label=clip_label(t, lvl))
        print("FRAMES", len(frames), "valid", sum(f is not None for f in frames))
        frames = [f for f in frames if f is not None]
        h, w = frames[0].shape[:2]
        path = os.path.join(args.out, f"{t}_{lvl}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                               "-r", str(int(round(1 / u.step_dt))), "-i", "-", "-vf", "scale=960:-2", "-c:v", "libx264",
                               "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        for f in frames:
            ff.stdin.write(np.ascontiguousarray(f[:, :, :3]).astype(np.uint8).tobytes())
        ff.stdin.close()
        ff.wait()
        print("VIDEO", path)

env.close()
app.close()
