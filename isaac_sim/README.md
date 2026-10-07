# Traquad track tests in Isaac Sim

## Isaac asset (`assets/traquad/traquad.usda`)

Instanceable USD of the current traquad model (roller wheels, real masses 7.8 kg, left/right symmetric),
ready for Isaac Sim / Isaac Lab (`sim_utils.UsdFileCfg(usd_path=...)`; used by `../open_traquad.py`).

- Built from the xacro by `./make_isaac_asset.sh [roller_damping] [roller_friction]`
  (xacro -> URDF -> URDF importer -> `finalize_usd.py`); rebuild it whenever the xacro changes.
- Meshes are stored once in `payloads/geometries.usd` and referenced as instances (`instanceable = true`).
- Physics variant `physx` selected, floating base, articulation self-collisions off.
- Default drives written in the file: HFE PD Kp 100 / Kd 10, max 10 Nm; ankles passive damping 0.01 Nm s/rad, end
  stops +-30 deg around the flat pose of the default stance (HFE +-1.47, ankle +-0.1008); **rollers passive, damping
  1e-4 Nm s/rad, dry friction 0.06 Nm**; wheels velocity drive with the track motor split over the 4 wheels of a track:
  **1.5 Nm and 2 Nm s/rad per track at r = 15 mm (`--track_max_torque`, `--track_damping`)**, i.e. 0.375 Nm and
  0.5 Nm s/rad per wheel, armature 0.001 (1.5 Nm = 100 N of belt force).
- Ankle control (not in the file, added by the controllers: `ankle_control.py` here, the track action in Isaac Lab,
  `../open_traquad.py`): the drive sprocket (radius r_s = 25 mm, placeholder) is coaxial with the ankle. Track touching
  the ground: the ankle gets the torque the sprocket puts on the track, (r_s / r_wheel) x the drive torque of the track
  wheels (the simulated motor drives the wheels, so without it the traction pitches the track with F h instead of
  F (h - r_s) and the tracks tip on slopes the real robot climbs). Track in the air for more than 50 ms: PI velocity
  control of the ankle towards v_track / r_s (Kp 0.2 Nm s/rad, Ki 2 Nm/rad, max 0.05 Nm, anti-windup), so the frame
  turns with the sprocket, however slow the command, until an end stop; with no command it holds its angle.
- Track mass: body 60% with its CoM placed so that the whole track has its CoM below the ankle axis when flat.
- Wheel-ground friction 0.75 in the tests (open_traquad.py: ground 1.0 averaged with the robot default 0.5;
  track_test.py `--mu 0.75`; Gazebo `wheel_mu` 0.75). Track bodies collide as boxes, contact offset 3 mm on wheels,
  rollers and track bodies. Roller dry friction: `physxJointAxis:angular:static/dynamicFrictionEffort`. Isaac Lab
  actuators override the joints they match (HFE, ankles, wheels); the rollers keep the values of the file.
- Not in the file: the contact material (friction with the ground) and the ground itself.
- **Needs Isaac Sim 6.1** (and Isaac Lab 3.0 for `../open_traquad.py`): the links are nested rigid bodies,
  which Isaac Sim 5.x does not parse (`CreateJoint - no bodies defined at body0 and body1` on every joint
  below the HFE).

## Setup (conda)

```bash
conda create -n env_isaaclab3 python=3.12 && conda activate env_isaaclab3
pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
# from the repo root: Isaac Lab 3.0 in ./isaaclab_traquad with the TraQuad tasks linked in (see below)
./isaaclab_overlay/setup_isaaclab.sh
cd isaaclab_traquad && ./isaaclab.sh -i && cd ..
python open_traquad.py --demo --viz kit
```

The TraQuad code for Isaac Lab lives in `../isaaclab_overlay/` (versioned in this repository).
`setup_isaaclab.sh` clones Isaac Lab 3.0 (`release/3.0.0` of github.com/isaac-sim/IsaacLab, commit `7aba91f`) into
`../isaaclab_traquad` (git-ignored, its own git repository) and links these files into it:

- `source/isaaclab_assets/isaaclab_assets/robots/traquad.py`: robot config, uses `assets/traquad/traquad.usda`
  (found through the layout `traquad_simulation/isaaclab_traquad/...`, or the `TRAQUAD_USD` variable)
- `source/isaaclab_tasks/isaaclab_tasks/contrib/traquad/`: tasks `Isaac-Velocity-Flat-TraQuad` and
  `Isaac-Velocity-Rough-TraQuad` (PhysX backend), with their MDP terms (one velocity action per track, wheel rewards,
  wheel-ground friction randomized per robot (static 0.6-1.3, dynamic 0.8-1.0 x static, same on all its colliders),
  roller dry friction fixed at 0.06 N m (a property of the robot); commands v and w in [-1, 1], 1 unit of track action
  = 1.3 m/s;
  leg power penalty and per-track stuck recovery reward, default stance HFE +-1.47, ankles driven by the track action
  in the air) and RSL-RL agents (rough: 15000 iterations, flat: 5000)
- `Isaac-Velocity-Mixed-TraQuad` (`mixed_env_cfg.py`, the training task): the rough task on flat 15%, rough noise
  1-6 cm 15%, stairs and inverted stairs 2-10 cm 15% + 15%, boxes 2-10 cm 10%, pyramid and inverted pyramid slopes
  0-40 deg 15% + 15% (5 cm cells, slope threshold 0.9: 40 deg slopes stay smooth, steps stay vertical); one friction
  per robot for the whole training, static 0.6-1.3, dynamic 0.3-1.0 (<= static); 10% standing commands; rewards:
  lin_vel_z_l2 -0.02, stand_still_wheels (wheel speeds at zero command) -0.002, track_slip (mean belt speed - base
  speed) -0.1. Logs in `logs/rsl_rl/traquad_mixed`.

On this machine `../isaaclab_traquad` also keeps `logs/`, `outputs/`, `isaac_model/` (runs of the Isaac Lab 2.x fork)
and `_archive/isaaclab_traquad_2x_fork.tar.gz` (its sources: old TraQuad and OmniQuad tasks).

```bash
cd ../isaaclab_traquad
export OMNI_KIT_ACCEPT_EULA=YES
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Mixed-TraQuad   # headless, 1024 envs (~4.5 GB GPU), 15000 it.
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-TraQuad --max_iterations 10000 --seed 1
isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Mixed-TraQuad --viz kit
```

To open the asset by hand: run `isaacsim` in the env, *File > Open* `assets/traquad/traquad.usda`
(or drag it into a stage), add a ground plane and press Play.

Same maneuver and metrics as the Gazebo tests, run in Isaac Sim 6.1 (PhysX, 1 ms step, headless).

## Steps

```bash
cd isaac_sim
./make_urdf.sh /tmp/traquad.urdf                    # current model (roller wheels); needs the ROS container running
./import_urdf.sh /tmp/traquad.urdf /tmp/traquad_usd  # URDF -> USD
./isaac.sh track_test.py --usd /tmp/traquad_usd/traquad/traquad.usda --out /tmp/run.csv --info   # quick check
./batch.sh /tmp/traquad_usd/traquad/traquad.usda     # damping sweep, results in ./results

# settings of ../open_traquad.py (Isaac Lab training setup), without Isaac Lab
./isaac.sh open_traquad_replica.py --usd /tmp/traquad_usd/traquad/traquad.usda --roller_damping 1e-4
# video with a follow camera (PNG frames named by simulated time, ~60 per simulated second)
./isaac.sh video_replica.py --usd /tmp/traquad_usd/traquad/traquad.usda --out /tmp/frames --roller_damping 1e-4
ffmpeg -framerate 60 -pattern_type glob -i '/tmp/frames/*.png' -c:v libx264 -pix_fmt yuv420p video.mp4
```

`ISAAC_SIM_DIR` sets the Isaac Sim install path (default `~/Downloads/isaac-sim-standalone-6.1.0-linux-x86_64`);
if it does not exist, `isaac.sh` uses the python of the active environment when it has `isaacsim>=6` (conda setup above).

## Test setup (`track_test.py`)

- PhysX, 1 ms step (`--dt`), TGS solver, CPU; ground is a collision plane; isotropic friction `--mu` (0.75) on ground
  and robot.
- HFE: PD Kp = 100, Kd = 10, max 10 Nm, stance `--hfe` (1.47, default stance; 1.13 in the earlier tests). Wheels:
  velocity drive and torque limit from the USD (`--wheel_max` per wheel). Ankles: passive damping 0.01 plus the
  ankle controller in the air (`--no_coupling`: passive only). Rollers: damping `--damping`, dry friction
  `--roller_friction` (default: USD).
- Initial state: legs at target, tracks flat.
- Maneuver: 3 s settle, 8 s rotation in place (ω = 0.5 rad/s), 2 s stop, 10 s turn (v = 0.2 m/s, ω = 0.5 rad/s), 2 s
  stop, 5 s straight. Track speeds from r = 0.015 m, B = 0.395 m. Besides yaw, drift and pose error it reports the
  HFE deviation, the time with an ankle on an end stop, and per phase the ankle angles, the time the HFE are saturated
  and the track motor torques.

## Scripts

| Script | What it does |
|---|---|
| `track_test.py` | standard maneuver of the Gazebo tests, metrics + CSV |
| `open_traquad_replica.py` | stance, drives, dt, friction and command sequence of `open_traquad.py`; prints measured vs commanded velocity per segment |
| `video_replica.py` | same setup as the replica, renders a follow-camera video with command and measured velocity on each frame |
| `lateral_force.py` / `lateral_batch.sh` | robot parked, constant lateral force on the base for 3 s: lateral speed and displacement (force x roller damping) |
| `friction_batch.sh` | for each roller dry friction: lateral push matrix + rotation sequence |
| `ramp_test.py` / `ramp_batch.sh` | robot parked across a side slope (tilted gravity), mass set with `--mass`: holds or slides (friction x slope) |
| `ramp_video.py` | two robots on a real inclined ramp with different roller dry friction |
| `ankle_swing.py` | robot suspended, hips locked: tracks released from a tilt and/or driven by a sequence of track speeds (`--sequence "dur:v,..."`), side camera video, ankle angles to CSV |
| `ankle_control.py` | ankle model shared by the scripts (passive on the ground, PI velocity control in the air, physics-step callback) |

Roller dry friction (`--roller_friction`, Nm) is set at runtime with
`set_dof_friction_properties` (static = dynamic) on the 96 roller joints; in URDF terms it is
`<dynamics friction="..."/>` on the roller joints. A roller turns only when the lateral contact force
exceeds tau / r_roller, so the robot holds lateral forces up to about 12 * tau / 0.008 = 1500 * tau N.

## Pitfalls found

- Standalone Isaac Sim: run it through `isaac.sh`, an active conda environment or the system `libcudart` 12.0
  makes it exit at startup. The pip install in its own conda env (setup above) has no such problem.
- The imported USD has a `Physics` variant set with no default: `track_test.py` selects `physx`,
  otherwise the robot has no physics.
- Use a collision plane as ground: a thin large box makes the cylinder-wheel contacts explode.
- Set the initial ankle angles: starting at 0 (outside the ankle limits) leaves the hind tracks tilted.
- Camera lens: set `focalLength` / aperture on the USD camera prim (in mm); `RtxCamera.camera.set_focal_lengths` uses other units.
- Track width in the controller: 0.395 m (contact lines at y = ±0.1975), not 0.35.
- Rigid track wheels do not turn in PhysX. With plain cylinder wheels (`TRACK_MODEL=cylinder`, 2026-10) the robot
  goes straight cleanly (0.200 m/s, vertical jitter 0.7 m/s^2 at 1 ms vs 3.4 with rollers, 0.03 at 5 ms) but rotates
  0% in place and in a curve (7% / 4% with mu 0.75 and 40 Nm per track, 2026-10-06). Not the cause: torque limit (0.3 or 10 Nm), mimic joints, polygonal cylinders
  (`collisionApproximateCylinders` is False: analytic), sphere instead of cylinder colliders, box vs mesh track
  bodies, time step 1-5 ms, friction correlation distance. Contacts are far from the center (a = 0.32 m) compared
  with the half track width (b = 0.20 m), and lateral friction holds the robot; ideal Coulomb friction would give
  ~28% (b^2 / (a^2 + b^2)), the rollers give 24%. The rollers are what makes the lateral slip possible in PhysX.
- Mimic joints in PhysX: the URDF importer writes `NewtonMimicAPI`, which PhysX accepts only if leader and follower
  joints have finite limits (continuous wheels have none): `finalize_usd.py` sets +-1e7 deg.
- `PhysxSurfaceVelocityAPI` (belt as surface velocity, `sv_test.py`): read only at simulation start (runtime USD
  changes are ignored), not applied with GPU dynamics, and the robot still does not turn (9% in place, 0% in a curve).
- `track_test.py` torque demand is the projected joint force of the driven wheels: an approximation of the drive
  torque.
- Isaac Lab with many robots: PhysX GPU articulations corrupt the articulation state with more than 32 instances,
  more than 64 links (the roller model has 121) and `gpu_max_num_partitions > 1` (isaac-sim/IsaacLab#8121): robots
  launched at tens of m/s and NaN observations even with zero actions. The TraQuad tasks set
  `gpu_max_num_partitions = 1` (no measurable slowdown: ~5200 steps/s with 1024 envs).
- Stability of the roller model (track_test.py, final asset, mu 0.75, 10 Nm per wheel, 2026-10-06): no NaN at 1 and
  5 ms; yaw 90% in place and 91% / 89% in the turn, straight 0.197-0.198 m/s, lateral drift in the turn 0.124 / 0.106
  m/s; roll/pitch rate up to 0.62 / 0.54 rad/s, ankles up to 6.0 / 10.0 rad/s, rollers up to 174 / 187 rad/s.
- Roller dry friction is a trade-off (5 ms, damping 1e-4, 2026-10-06). Driving: tau 0 -> 96% / 96% yaw, drift 0.013
  m/s, final pose error 0.12 m; 0.02 -> 93% / 92%, 0.027 m/s; 0.03 -> 0.078 m/s; 0.06 -> 90% / 89%, 0.106 m/s, 0.74 m,
  more vibration; 0.1 -> 75% / 71%. Parked (7.8 kg): tau 0 slides under any push or slope; 0.02 holds 20 N and 20 deg;
  0.06 holds 40 N and 20 deg, creeps a few mm/s at 25-35 deg; every model slides at 40 deg (mu 0.75). Roller damping
  matters little below 1e-4 (friction dominates); 1e-2 halves the rotation.
- Tipping under traction (2026-10-07): the traction F acts below the ankle pivot (h = 73 mm). With the motor torque
  inside the track (wheel motors) the pitch moment is F h and the tracks tipped onto their stops on grippy ground
  (from mu 0.85 on flat ground) and on every 35 deg slope, whatever the motor torque (3 or 40 Nm too): the real robot
  climbs 35 deg on wooden boards. With the sprocket torque on the ankle (r_s = 25 mm) the moment is F (h - r_s): no
  tipping on flat ground up to mu 1.0 and climbs of 35 deg at mu 0.75. The track motor (1.5 Nm = 100 N per track) is
  far from the limit (35 deg needs 11 N per track); 40 Nm gave torque spikes at command steps. At stance 1.13 the hips
  need 3.2-3.6 Nm to stand (1.5-1.6 at 1.47). Climb tests: `track_test.py --slope deg` (gravity tilted progressively,
  `--slope_ramp`, default 2 s; at once it tips the tracks), `video_replica.py --slope deg`.
- Lateral grip when parked, tau 0.06 vs 0.02 against the wheel-ground friction mu (`ramp_test.py` / `lateral_force.py
  --mu`, which also print the roller speed, 2026-10-07). With tau 0.06 the rollers never turn: the robot slides on the
  ground as a block, at tan(slope) > mu (mu 0.3 / 0.5 / 0.75 / 1.0: slides at 20 / 30 / 40 / 40-45 deg) and at a push of
  about mu*m*g (30 / 40 / 50 / 60 N). With tau 0.02 the rollers give way at about 30 N or 20-25 deg whatever mu is.
  The rollers stay locked while tau > mu*m*g*r_roller/N_contacts = 0.052*mu (7.8 kg, 12 rollers): tau is a property of
  the robot, the grip comes from mu.
- The scripts of this folder use the current model: HFE Kd 10, passive ankles (damping 0.05), real mass 7.8 kg;
  `--roller_friction` is always applied (default 0.06, the asset value; 0 removes it). `video_replica.py` also
  handles the cylinder model (only the driven wheels are commanded) and logs velocities with `--csv`.
