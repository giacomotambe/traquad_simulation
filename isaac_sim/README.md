# TraQuad in Isaac Sim

The TraQuad (Mulinex on tracks, 8.0 kg) simulated in Isaac Sim 6.1 (PhysX) and Isaac Lab 3.0, with its parameters
identified on the real runs of 22-23 October 2026 (robot bags + Qualisys, github.com/giacomotambe/traquad_bags).
What changed and when: [Changelog](#changelog).

## Contents

| Path | What it is |
|---|---|
| `assets/traquad/traquad.usda` | the robot asset used everywhere (Isaac Sim scripts, `../open_traquad.py`, Isaac Lab) |
| `traquad_real_controller.py` | numpy replica of the real controllers: NUC joystick with the R2 posture, SBC omni_controller (twist clamp, 2 Hz low-pass, TraquadIK, wheel compensation), moteus drives |
| `real_replay.py` | replays a real run in Isaac Sim through the controller replica, several robots in parallel with different parameters (`--grid`) |
| `real_id/` | `bag2npz.py` (bags -> npz), `runs.py` (Qualisys / robot clock alignment), `compare.py` (sim vs real, plots, 10 s window trajectory error), `score.py` (ranks a parameter grid), `segments.py`, `missions.py`, `ghost.py` (motion of the real robot for the videos) |
| `render_replay.py` | video of a replay (`real_replay.py --frames`) with the real robot next to the simulated one |
| `finalize_usd.py` | writes drives, frictions and the hip offset into the asset (rerun it to change them, no ROS needed) |
| `make_isaac_asset.sh`, `make_urdf.sh`, `import_urdf.*` | rebuild the asset from the xacro (`../src/mulinex_description/urdf/traquad.xacro`) |
| `ankle_control.py` | ankle model shared by the scripts (passive on the ground, PI towards the sprocket speed in the air) |
| `track_test.py` | open-loop maneuver and climb test (`--slope`); plain 0.4 m differential, no compensation |
| `lateral_force.py`, `ramp_test.py` | robot parked: side push, side slope |
| `ankle_swing.py`, `video_replica.py` | robot suspended (ankle tests), follow-camera video |
| `isaac.sh` | runs a script headless with the conda env or the standalone Isaac Sim |

## The model

Each track is 4 wheels of 15 mm radius under a track body (collision box); wheels 1-3 carry 8 passive sphere rollers
(omni-wheel like: grip along the track, roll sideways against damping and dry friction), wheel 4 is the raised idler.
All 4 wheels are driven by the track motor, referred from the 45 mm sprocket of the real track.

| Parameter | Value | Source |
|---|---|---|
| Hip PD | Kp 60 N m/rad, Kd 0.6 N m s/rad, limit 10 N m | robot code (mulsbc_ws traquad8_dji.xacro); the limit is a placeholder |
| Hip zero | joint frame shifted by c = -0.05 rad: geometric leg angle = encoder angle + c | identified (R2 vs no R2 turning); check on the robot |
| Track speed loop | Kd 0.9 N m s/rad at the track joint, Kp 0: PhysX velocity drive, 0.025 per wheel | robot code; the drive must be implicit (see below) |
| Sprocket radius | 45 mm (belt per rad of the track joint) | wheel_rad, Qualisys straight runs 2026-10-22 |
| Track torque limit | 0.83 x hip limit = 8.3 N m at the joint (0.69 N m per wheel) | same actuator as the hips: 40 A x gear 15 vs 80 A x gear 9 |
| Internal track friction | 1.16 N m + 0.013 N m s/rad at the track joint, as joint friction of the wheels | identified: drive torque on the straight runs |
| Roller dry friction / damping | 0.10 N m / 1e-4 N m s/rad | identified with mu (trajectory fit) |
| Wheel-ground friction | mu 0.45 (static = dynamic) on the lab mat | identified (trajectory fit); the sliding test gave 0.6 |
| Ankles | on the ground passive, damping 0.01 N m s/rad, stops +-30 deg, sprocket torque off; in the air driven by the controller of its track with the same speed command (Kd 0.9 as drive, integral ki 3 / +-1 N m) | robot (frame turns with the sprocket); the passive damping is a model choice |
| Mass | 8.0 kg (weighed) | robot |

The friction of the ground is not in the asset: the scripts bind it (`--mu`, default 0.45); Isaac Lab randomizes it.

## Replaying a real run

```bash
# once: the bags of github.com/giacomotambe/traquad_bags (release), pip install rosbags
python real_id/bag2npz.py <extracted run dir> run.npz <mulsbc_ws>/src/pi3hat/pi3hat_moteus_int_msgs/msg
./isaac.sh real_replay.py --data run.npz --preset sim_ident --t0 745 --t1 860 --r2 747.1-900 --out sim.npz
python real_id/compare.py run.npz sim.npz --plot cmp.png      # velocities, track speeds/torques, hips, trajectory
python real_id/score.py run.npz:sim.npz                       # 10 s window trajectory error, y_eff, RMS errors
```

- `--preset`: the controller settings of the run (`sim_ident` no compensation, `deficit` IK deficit compensation,
  `lowlevel` wheel compensation, `current` = HEAD of mulsbc_ws / mulinex_app).
- `--r2 a-b,...`: when R2 was on (robot time). `--wheel_source direct --hfe_source log` replays the identification
  missions (direct track commands, logged hip references).
- `--grid "mu=0.45,0.6;roller_friction=0.08,0.1"`: one robot per combination, side by side (12 robots ~ 8 s of wall
  per simulated second on this PC).

The replica reproduces the commands logged on the robot within 0.002 rad on the hips and 0.1-0.2 rad/s on the tracks.
The track controller of the robot is the moteus Kd 0.9 plus the omni_controller wheel compensation (integral ki 3
N m/rad, +-1 N m, and friction feedforward 1.4 + 0.017 |w| N m): presets `lowlevel` and `current`; the runs recorded
before it (`sim_ident`, `deficit`) are replayed without it, as they were driven.

Videos with the real robot next to the simulated one (orange: posed from the Qualisys and the encoders):

```bash
python real_id/ghost.py run.npz ghost.npz --marker 0.05 -0.03 --t0 20 --t1 100    # marker offset from score.py
./isaac.sh real_replay.py --data run.npz --preset deficit --t0 28 --t1 90 --r2 77.2-200 --frames frames.npz --out sim.npz
./isaac.sh render_replay.py --frames frames.npz --ghost ghost.npz --out frames_dir --title "..."
ffmpeg -framerate 30 -i frames_dir/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 26 video.mp4
```

## Validation

Replay of the teleop runs of 2026-10-23 with the asset as it is (mu 0.45), against the Qualisys. Window error: mean
distance (and heading difference) between real and simulated robot at the end of 10 s windows started from the same
pose; ratios are least-squares slopes of the simulated against the measured velocity.

| Run (controller on the robot) | Window error | Yaw rate sim / real | Forward speed sim / real | RMS error wz / vx |
|---|---|---|---|---|
| R2, no compensation | 0.14 m, 8 deg | 1.05 | 0.95 | 0.052 rad/s, 0.043 m/s |
| without R2 then R2, IK deficit compensation | 0.12 m, 8 deg | 1.07 | 0.94 | 0.077 rad/s, 0.061 m/s |
| R2, wheel compensation (current robot code) | 0.17 m, 9 deg | 1.00 | 0.95 | 0.054 rad/s, 0.045 m/s |

Track speed loss and drive torque match too (1.71 vs 1.74 rad/s, 1.54 vs 1.55 N m in the first run). Over a whole
run (2 minutes open loop) the end poses drift apart by 0.5-1.4 m and 50-70 deg: small velocity errors integrate.

### Slopes and side pushes (2026-10-27)

Climb (`real_replay.py --slope deg`, robot controllers, 0.3 m/s for 10 s then stopped): on the mat (mu 0.45) 0.25 /
0.22 m/s at 15 / 20 deg, slides down from 25 deg (tan 25 deg > 0.45); at mu 0.6 0.27 / 0.25 / 0.21 / 0.12 m/s at 15 /
20 / 25 / 30 deg; at mu 0.8 up to 25 deg, fails at 30 deg (tracks tipped onto their stops); nothing climbs 35 deg.
Side push on the parked robot (`lateral_force.py`, 3 s): holds 30 N on the mat (limit mu m g = 35 N), 40 N at mu
0.6-0.8, then yields by turning (50 N) and slides (60 N).

## Training (Isaac Lab)

```bash
./isaaclab_overlay/setup_isaaclab.sh             # from the repo root: Isaac Lab 3.0 in ./isaaclab_traquad + TraQuad
cd isaaclab_traquad && ./isaaclab.sh -i
export OMNI_KIT_ACCEPT_EULA=YES
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Mixed-TraQuad   # 1024 envs (~4.5 GB GPU)
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Mixed-TraQuad --viz kit
```

The robot config (`isaaclab_overlay/.../robots/traquad.py`) uses the same asset and the same track motor (45 mm
sprocket, internal friction as actuator friction, no sprocket torque). The tasks command the track speeds directly
(1 action unit = 1.3 m/s) and randomize the ground friction per robot (static 0.4-1.1, since 2026-10-27; it was 0.6-1.3):
the range covers the identified mat value, 0.45.

## Setup

```bash
conda create -n env_isaaclab3 python=3.12 && conda activate env_isaaclab3
pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
```

`isaac.sh` uses the standalone Isaac Sim in `ISAAC_SIM_DIR` if present, otherwise the active conda env. The asset needs
Isaac Sim 6.1 (nested rigid bodies are not parsed by 5.x). Rebuilding the asset from the xacro needs `xacro`
(`pip install xacro`, or the ROS container); `finalize_usd.py` alone does not.

## Known limits

- **Slopes**: on grippy ground the model fails from 30 deg because the traction tips the passive tracks onto their
  stops; the real robot climbs 35 deg on wood. With the sprocket torque (r_s 45 mm) the tracks tip on flat ground.
  The moment the real track sees is unknown: a side video of the ankles while climbing would settle it.
- **Forward creep while turning**: on the robot the track driving backward slips about twice as much as the forward
  one, so the robot creeps forward (the IK compensates it). The model is symmetric. A phenomenological fix
  (`real_replay.py --grid slip_bwd=...`) reproduces the creep but does not improve the trajectories, so it is off.
- **Other surfaces**: everything was fitted on the lab mat; the IK of the robot too.
- **Turning in place at fixed shoulder angle** (mission t1, direct track commands): y_eff +7% / +16% / +20% / -1% at
  1.05 / 1.3 / 1.6 / 1.85 rad; the fit favours the teleop runs.

## Pitfalls

- The track speed loop must be the PhysX velocity drive: a loop computed in Python each step (explicit effort) makes
  the roller contacts slip (robot at 25-60% of the belt speed at 0.5-1 m/s; 98% with the implicit drive).
- Internal friction as PhysX joint friction works only together with the implicit drive.
- The sprocket torque model of `ankle_control.py` (factor `SPROCKET_TORQUE_FACTOR`, 0) tips the tracks with r_s 45 mm.
- Isaac Lab with many robots: PhysX GPU articulations with more than 64 links break with `gpu_max_num_partitions > 1`
  (isaac-sim/IsaacLab#8121): the TraQuad tasks set it to 1.
- The imported USD has a `Physics` variant set with no default: the scripts select `physx`.
- Ground: a collision plane (a thin box makes the wheel contacts explode).
- Several Isaac processes at once fill the 8 GB GPU with renderers (headless too): at most 4 in parallel.

## Changelog

**2026-10-27 (afternoon)**
- Ankle in the air: driven by the controller of its track with the same speed command (track Kd as implicit drive,
  wheel-compensation integral as effort) in `ankle_control.py`, `real_replay.py` and the Isaac Lab track action (ankle
  actuator now implicit); it was a weak PI (Kp 0.2, Ki 2, max 0.05 N m). Checked with `ankle_swing.py` (the frame turns
  at v / r_s, holds at zero command) and an Isaac Lab smoke run; on flat ground the tracks never leave it.
- `real_replay.py --slope` (climb through the robot controllers) and `--frames`; `render_replay.py` and
  `real_id/ghost.py` for videos with the real robot.

**2026-10-27**
- Trajectory metric: end error of 10 s windows from the real start pose (`compare.window_errors`), the Qualisys marker
  offset removed by least squares (it comes out at about +6 cm, -4 cm in every run).
- mu identified at 0.45 with roller dry friction 0.10 (were 0.6 and 0.075); asset, scripts and `open_traquad.py`
  updated. Tested and left off: centre of mass shift (no effect on the creep), direction-dependent sprocket torque
  (bistable: tips the tracks), direction-dependent slip (creep yes, trajectories no better).
- Cleanup: removed the track model variants (cylinders, sphere pairs, gimbal, barrel rollers, other roller sizes)
  from the xacro, the assets, `finalize_usd.py`, `open_traquad.py` and Isaac Lab (`TRAQUAD_CYLINDER_CFG`); removed the
  scripts of past experiments (`sv_test.py`, `gimbal_reset.py`, `open_traquad_replica.py`, `ramp_video.py`, batch
  scripts); old results and videos archived in `../isaaclab_traquad/_archive/isaac_sim_results_2026-10.tar.gz`.
  The default URDF is unchanged (checked by diff).

**2026-10-26**
- Controller replica (`traquad_real_controller.py`) and replay of the real runs (`real_replay.py`, `real_id/`).
- Sprocket radius 28 -> 45 mm everywhere (track gain 0.065 -> 0.025 per wheel, limit 1.12 -> 0.69 N m per wheel).
- Internal track friction 1.16 + 0.013 |w| N m added (asset and Isaac Lab actuator).
- Hip zero offset -0.05 rad in the asset (R2 turns better than without, as on the robot).
- Sprocket torque on the ankle off (`SPROCKET_TORQUE_FACTOR` 0 in `ankle_control.py`, `open_traquad.py`, Isaac Lab).

**2026-10-20/21** (before the real data): real hip and track gains, mass 8.0 kg, track torque tied to the hip limit,
mu 0.6 from a sliding test.

**2026-10-05/09**: roller wheel model chosen over cylinders, sphere pairs, gimbal and barrel rollers (rigid wheels do
not turn in PhysX); Isaac Sim 6.1 / Isaac Lab 3.0.
