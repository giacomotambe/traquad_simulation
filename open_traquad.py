"""Spawn the TraQuad robot on a flat ground and drive it with a skid-steer (tank) command.

Each track is simulated by its 4 wheels per leg (16 wheels in total). A body twist command
(linear velocity v along x, yaw rate w) is converted into left/right track speeds and then
into wheel angular velocities:

    v_left  = v - w * B / 2
    v_right = v + w * B / 2
    omega_wheel = v_side / r

The wheels get the wheel compensation of the real track controller (omni_controller wheel_compensation, enabled on the
robot) on top of their velocity drive, and the ankles the model of the Isaac Lab tasks (passive on the ground, driven
with the sprocket by the track controller in the air): same law as isaac_sim/ankle_control.py.

Requires Isaac Lab 3.0 with Isaac Sim 6.1 (PhysX backend).

Usage (conda env with Isaac Lab 3.0):
    python open_traquad.py --lin_vel 0.3 --ang_vel 0.0 --viz kit
    python open_traquad.py --demo --viz kit
"""

import argparse
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="TraQuad on flat ground with track velocity commands.")
parser.add_argument("--lin_vel", type=float, default=0.3, help="Forward velocity command [m/s].")
parser.add_argument("--ang_vel", type=float, default=0.0, help="Yaw rate command [rad/s].")
parser.add_argument("--demo", action="store_true", help="Cycle through a sequence of (v, w) commands.")
parser.add_argument(
    "--rigid_legs", action="store_true", help="Hold HFE and ankle joints at the stance with very stiff drives."
)
parser.add_argument(
    "--wheel_friction",
    type=float,
    default=0.45,
    help="Static friction of the track wheels, min with the ground (default 0.45: identified on the lab rubber mat with "
    "the Qualisys runs; 0 keeps the robot default material 0.5, averaged with the ground).",
)
parser.add_argument(
    "--wheel_dynamic_friction",
    type=float,
    default=None,
    help="Dynamic (sliding) friction of the track wheels (min with the ground; default: same as the static one).",
)
parser.add_argument("--ground_friction", type=float, default=0.45, help="Friction of the ground plane.")
parser.add_argument(
    "--track_max_torque",
    type=float,
    default=10.0 * (40.0 * 15.0) / (80.0 * 9.0) * 0.015 / 0.045,
    help="Torque limit of a track motor [Nm at r = 15 mm]: same actuator as the hips (10 Nm), 40 A x gear 15 against "
    "80 A x gear 9, i.e. 0.83 x 10 Nm at the 45 mm sprocket (2.78 Nm = 185 N of belt force).",
)
parser.add_argument(
    "--ankle_stiffness",
    type=float,
    default=0.0,
    help="Ankle PD stiffness [Nm/rad] (0: ankle model of the asset and the RL tasks, driven by the track motor).",
)
parser.add_argument(
    "--no_wheel_comp", action="store_true", help="Moteus Kd only on the tracks, without the wheel compensation."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import quat_apply_inverse


# instanceable asset of this repository (rebuild with isaac_sim/make_isaac_asset.sh); TRAQUAD_USD overrides it
USD_PATH = os.environ.get(
    "TRAQUAD_USD",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_sim", "assets", "traquad", "traquad.usda"),
)

# Driven wheels (all 4 of each track) and their velocity drive (damping = velocity gain). Track motor (same values as
# isaac_sim/finalize_usd.py):
# torque limit --track_max_torque and the velocity gain of the real track joint (moteus Kd 0.9 N m s/rad at the 45 mm
# sprocket, Kp 0, mulsbc_ws traquad8_dji.xacro) referred to the 15 mm wheels, split over the driven wheels of the track
TRACK_MAX_TORQUE, TRACK_DAMPING = args_cli.track_max_torque, 0.9 * (0.015 / 0.045) ** 2
WHEEL_JOINTS, DRIVEN_PER_TRACK = "joint_wheel_.*_", 4
WHEEL_MAX_TORQUE, WHEEL_DAMPING = TRACK_MAX_TORQUE / DRIVEN_PER_TRACK, TRACK_DAMPING / DRIVEN_PER_TRACK

# Track geometry (from traquad.urdf)
WHEEL_RADIUS = 0.015  # [m]
# track width of the differential IK of the real robot (omni_controller track_width, default 0.4 m; not set in
# traquad8_dji.yaml). The simulated contact lines are at y = +-0.1975 (wheel joints at +-0.175, 45 mm wide wheels)
TRACK_WIDTH = 0.4
# Skid-steer slip compensation: effective width > geometric width (tune if turning is too slow/fast)
TRACK_WIDTH_FACTOR = 1.0
# Ankle (same model as isaaclab_assets/robots/traquad.py and isaac_sim/ankle_control.py): the drive sprocket is coaxial
# with the ankle. On the ground the ankle is passive (ANKLE_PASSIVE_DAMPING), plus the torque of the sprocket on the
# track, (SPROCKET_RADIUS / WHEEL_RADIUS) x the wheel drive torque, scaled by SPROCKET_TORQUE_FACTOR. With the track in
# the air the frame turns with the sprocket (v / SPROCKET_RADIUS) until an end stop, driven by the controller of its
# track: the moteus Kd (implicit velocity drive of the ankle) and the integral of the wheel compensation as effort,
# reset when the track touches the ground. Contact is taken from the height of the wheels above the flat ground of this
# script.
SPROCKET_RADIUS = 0.045  # [m], wheel_rad of the real robot IK (mulsbc_ws traquad8_dji.yaml, Qualisys 2026-10-22)
# scale of the sprocket torque on the ground; off since 2026-10-26: with r_s 45 mm it tips the tracks onto their stops
SPROCKET_TORQUE_FACTOR = 0.0
ANKLE_PASSIVE_DAMPING = 0.01  # [N m s/rad]
ANKLE_SPEED_GAIN = 0.9  # [N m s/rad], moteus Kd of the track joint
# wheel compensation of the omni_controller (mulsbc_ws traquad8_dji.yaml), at the track joint (sprocket): friction
# feedforward sign(w) (FRICTION + VISCOUS |w|), faded below FADE, plus the integral of the track speed error (KI, +-I_MAX)
WHEEL_COMP_FRICTION = 1.4  # [N m]
WHEEL_COMP_VISCOUS = 0.017  # [N m s/rad]
WHEEL_COMP_FADE = 1.0  # [rad/s]
WHEEL_COMP_KI = 3.0  # [N m/rad]
WHEEL_COMP_I_MAX = 1.0  # [N m]
ANKLE_INTEGRAL_GAIN = WHEEL_COMP_KI  # [N m/rad]
ANKLE_MAX_TORQUE = WHEEL_COMP_I_MAX  # [N m]
CONTACT_MARGIN = 0.004  # [m], wheel centre below WHEEL_RADIUS + this: the track touches the ground
AIR_DELAY = 0.05  # [s], time without contact before the ankle controller takes over
# first-order low-pass on the twist command, as the omni_controller of the real robot (velocity_lpf, 2 Hz)
TWIST_LPF_HZ = 2.0

# Wheel joint axes point along -y (left) and +y (right) in the world frame,
# so a positive forward speed needs a negative spin on the left and positive on the right.
LEFT_SIGN = -1.0
RIGHT_SIGN = 1.0

# (duration [s], v [m/s], w [rad/s]) used with --demo
DEMO_SEQUENCE = [
    (4.0, 0.3, 0.0),
    (4.0, 0.0, 0.8),
    (4.0, 0.3, 0.5),
    (4.0, -0.3, 0.0),
    (4.0, 0.0, -0.8),
    (2.0, 0.0, 0.0),
]


# =====================================================================
# ROBOT CONFIGURATION
# =====================================================================

def make_robot_cfg(prim_path, position):

    return ArticulationCfg(
        prim_path=prim_path,

        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,

            activate_contact_sensors=True,

            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),

            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            # make sure the base is floating even if the USD was imported with a fixed base
            fix_root_link=False,
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            joint_pos={
                # |HFE| + |ankle| = pi/2 keeps wheels 1-3 of each track flat on the ground.
                # Largest HFE (ankle close to its limit) brings the tracks closest to the body center,
                # which makes skid steering easier.
                "LF_HFE": 1.47,
                "LH_HFE": -1.47,
                "RF_HFE": -1.47,
                "RH_HFE": 1.47,
                # ankle end stops (relative to the upper leg): +-30 deg around the flat pose, front in [-0.423, 0.624],
                # hind in [-0.624, 0.423]
                "body_.*_F_ankle": 0.10,
                "body_.*_H_ankle": -0.10,
                "joint_wheel_.*": 0.0,
            },
        ),

        actuators=make_actuators_cfg(),
    )


def make_actuators_cfg():

    if args_cli.rigid_legs:
        # very stiff drives with a large effort limit
        return {
            "legs_and_ankles": ImplicitActuatorCfg(
                joint_names_expr=[".*HFE", "body_.*_ankle"],
                joint_effort_limit=1000.0,
                joint_velocity_limit=100.0,
                stiffness=1.0e4,
                damping=100.0,
            ),
            "wheels": make_wheels_actuator_cfg(),
        }

    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HFE"],
            # hip motor limit raised from 5 N m (Gazebo value): the hips saturated in the first trainings
            joint_effort_limit=10.0,
            # real robot (mulsbc_ws traquad8_dji.xacro, joint units): moteus max_velocity 10 rad/s, Kp 60, Kd 0.6
            joint_velocity_limit=10.0,
            stiffness=60.0,
            damping=0.6,
        ),
        # ankles: coupled to the track motor by default (velocity drive towards the sprocket speed, as the asset and
        # the Isaac Lab tasks); --ankle_stiffness > 0 holds them with a PD instead
        "ankles": (
            ImplicitActuatorCfg(
                joint_names_expr=["body_.*_ankle"],
                joint_effort_limit=10.0,
                joint_velocity_limit=10.0,
                stiffness=args_cli.ankle_stiffness,
                damping=0.2,
            )
            if args_cli.ankle_stiffness > 0.0
            else ImplicitActuatorCfg(
                joint_names_expr=["body_.*_ankle"],
                # the integral is clipped to ANKLE_MAX_TORQUE below; the sprocket torque reaches 0.83 x 10 N m
                joint_effort_limit=10.0,
                joint_velocity_limit=100.0,
                stiffness=0.0,
                # passive damping on the ground, switched to ANKLE_SPEED_GAIN in the air by TrackController
                damping=ANKLE_PASSIVE_DAMPING,
            )
        ),
        "wheels": make_wheels_actuator_cfg(),
    }


def make_wheels_actuator_cfg():

    # velocity control: zero stiffness, damping acts as the velocity gain
    return ImplicitActuatorCfg(
        joint_names_expr=[f"{WHEEL_JOINTS}.*"],
        joint_effort_limit=WHEEL_MAX_TORQUE,
        joint_velocity_limit=200.0,
        stiffness=0.0,
        damping=WHEEL_DAMPING,
        # the wheels are tiny (r = 1.5 cm): armature improves the solver conditioning
        armature=0.001,
    )


# =====================================================================
# SCENE
# =====================================================================

def design_scene():

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.ground_friction,
            dynamic_friction=args_cli.ground_friction,
        ),
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.8, 0.8, 0.8),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = make_robot_cfg("/World/Traquad", (0.0, 0.0, 0.30))
    robot_cfg.spawn.func(robot_cfg.prim_path, robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    make_floating_base(robot_cfg.prim_path)
    if args_cli.wheel_friction > 0.0:
        set_wheel_friction(robot_cfg.prim_path, args_cli.wheel_friction, args_cli.wheel_dynamic_friction)
    # the prim is already spawned, so the Articulation must not spawn it again
    robot_cfg.spawn = None

    return Articulation(robot_cfg)


def make_floating_base(prim_path):
    """Move the articulation root from the world-fixed ``root_joint`` to the base link.

    The TraQuad USD was imported with a fixed base: the ArticulationRootAPI lives on a fixed joint
    to the world. Disabling that joint alone leaves the articulation without a valid root.
    """

    from pxr import PhysxSchema, UsdPhysics

    stage = sim_utils.get_current_stage()

    for prim in sim_utils.get_all_matching_child_prims(prim_path, lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI)):

        if not prim.IsA(UsdPhysics.Joint):
            continue

        joint = UsdPhysics.Joint(prim)
        targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
        if len(targets) == 0:
            continue
        base_prim = stage.GetPrimAtPath(targets[0])

        print(f"[INFO] Moving articulation root: {prim.GetPath()} -> {base_prim.GetPath()}")

        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        joint.GetJointEnabledAttr().Set(False)

        UsdPhysics.ArticulationRootAPI.Apply(base_prim)
        physx_api = PhysxSchema.PhysxArticulationAPI.Apply(base_prim)
        physx_api.GetEnabledSelfCollisionsAttr().Set(False)
        physx_api.GetSolverPositionIterationCountAttr().Set(16)
        physx_api.GetSolverVelocityIterationCountAttr().Set(4)


def set_wheel_friction(prim_path, friction, dynamic_friction=None):
    """Bind a physics material with the given friction to all the track wheels.

    The "min" combine mode has priority over the ground's "average", so the wheel-ground
    friction is min(wheel, ground). A dynamic friction lower than the static one lets the wheels
    grip while rolling along the track and slide more easily sideways, as real tracks do in a turn.
    """

    if dynamic_friction is None:
        dynamic_friction = friction

    material_path = "/World/Materials/WheelMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=dynamic_friction,
        friction_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)

    from pxr import UsdPhysics

    # colliders of the wheels (and of their rollers), at any depth: the links of the USD are nested
    colliders = sim_utils.get_all_matching_child_prims(
        prim_path, lambda p: p.HasAPI(UsdPhysics.CollisionAPI) and "/wheel_" in str(p.GetPath())
    )
    for collider in colliders:
        sim_utils.bind_physics_material(collider.GetPath(), material_path)

    print(f"[INFO] Wheel friction set to static {friction} / dynamic {dynamic_friction} on {len(colliders)} colliders")


# =====================================================================
# TRACK COMMANDS
# =====================================================================

def twist_to_wheel_velocities(robot, left_ids, right_ids, lin_vel, ang_vel):

    width = TRACK_WIDTH * TRACK_WIDTH_FACTOR

    v_left = lin_vel - ang_vel * width / 2.0
    v_right = lin_vel + ang_vel * width / 2.0

    joint_vel = torch.zeros_like(robot.data.joint_vel.torch)
    joint_vel[:, left_ids] = LEFT_SIGN * v_left / WHEEL_RADIUS
    joint_vel[:, right_ids] = RIGHT_SIGN * v_right / WHEEL_RADIUS
    return joint_vel


class TrackController:
    """Wheel compensation of the tracks and control of the ankles in the air (run every physics step)."""

    def __init__(self, robot, dt, ankles=True, wheel_comp=True):
        self.robot, self.dt, self.ankles, self.wheel_comp = robot, dt, ankles, wheel_comp
        self.tracks = []
        for side, track in (("left", "LEFT"), ("right", "RIGHT")):
            for fh in ("F", "H"):
                ankle_id = robot.find_joints(f"body_{side}_{fh}_ankle")[0][0]
                wheel_bodies = robot.find_bodies(f"wheel_[1-4]_{track}_{fh}")[0]
                wheel_joints = robot.find_joints(f"{WHEEL_JOINTS}{track}_{fh}")[0]
                self.tracks.append((ankle_id, wheel_bodies, side == "left", wheel_joints))
        self.ankle_ids = [a for a, _, _, _ in self.tracks]
        self.integral = torch.zeros(robot.num_instances, len(self.tracks), device=robot.device)
        self.air_time = torch.zeros_like(self.integral)
        self.was_in_air = torch.zeros_like(self.integral, dtype=torch.bool)
        self.comp_integral = torch.zeros_like(self.integral)

    def apply(self, v_left, v_right):
        # track speeds (ankle axes point along +y on both sides: the sprocket turns with v / SPROCKET_RADIUS)
        v = torch.tensor([v_left if left else v_right for _, _, left, _ in self.tracks], device=self.robot.device)
        if self.wheel_comp:
            self._wheel_compensation(v)
        if self.ankles:
            self._ankles(v / SPROCKET_RADIUS)

    def _wheel_compensation(self, v):
        # track joint (sprocket) speed and command, in the spin direction of the wheels
        ratio = WHEEL_RADIUS / SPROCKET_RADIUS
        joint_vel = self.robot.data.joint_vel.torch
        w_cmd = v / SPROCKET_RADIUS
        w = torch.stack([(LEFT_SIGN if left else RIGHT_SIGN) * joint_vel[:, wj].mean(dim=1) * ratio
                         for _, _, left, wj in self.tracks], 1)
        ff = torch.sign(w_cmd) * (WHEEL_COMP_FRICTION + WHEEL_COMP_VISCOUS * w_cmd.abs()) * (
            w_cmd.abs() / WHEEL_COMP_FADE).clamp(max=1.0)
        self.comp_integral = (self.comp_integral + WHEEL_COMP_KI * (w_cmd - w) * self.dt).clamp(-WHEEL_COMP_I_MAX,
                                                                                               WHEEL_COMP_I_MAX)
        tau = ff + self.comp_integral
        for j, (_, _, left, wj) in enumerate(self.tracks):
            tau_wheel = (LEFT_SIGN if left else RIGHT_SIGN) * tau[:, j:j + 1] * ratio / len(wj)
            self.robot.actuators.target_command.set_effort_index(value=tau_wheel.expand(-1, len(wj)), joint_ids=wj)

    def _ankles(self, target):
        z = self.robot.data.body_pos_w.torch[:, :, 2]
        off_ground = torch.stack(
            [z[:, w].min(dim=1).values > WHEEL_RADIUS + CONTACT_MARGIN for _, w, _, _ in self.tracks], 1
        )
        self.air_time = torch.where(off_ground, self.air_time + self.dt, 0.0)
        in_air = self.air_time > AIR_DELAY   # debounce
        target = target.expand_as(self.integral)
        err = target - self.robot.data.joint_vel.torch[:, self.ankle_ids]
        self.integral = torch.where(
            in_air, (self.integral + ANKLE_INTEGRAL_GAIN * err * self.dt).clamp(-ANKLE_MAX_TORQUE, ANKLE_MAX_TORQUE), 0.0
        )
        if (in_air != self.was_in_air).any():
            damping = torch.where(in_air, ANKLE_SPEED_GAIN, ANKLE_PASSIVE_DAMPING)
            self.robot.write_joint_damping_to_sim_index(damping=damping, joint_ids=self.ankle_ids)
            self.was_in_air = in_air.clone()
        self.robot.actuators.target_command.set_velocity_index(value=torch.where(in_air, target, 0.0),
                                                               joint_ids=self.ankle_ids)
        # on the ground: sprocket torque on the track. Wheel axes along -y (left) / +y (right), ankle axes along +y
        effort = self.robot.actuators.applied_effort.torch
        ratio = SPROCKET_TORQUE_FACTOR * SPROCKET_RADIUS / WHEEL_RADIUS
        sprocket = torch.stack(
            [(-1.0 if left else 1.0) * ratio * effort[:, wj].sum(dim=1) for _, _, left, wj in self.tracks], 1
        )
        self.robot.actuators.target_command.set_effort_index(value=torch.where(in_air, self.integral, sprocket),
                                                             joint_ids=self.ankle_ids)


def get_command(t):

    if not args_cli.demo:
        return args_cli.lin_vel, args_cli.ang_vel

    t = t % sum(d for d, _, _ in DEMO_SEQUENCE)
    for duration, v, w in DEMO_SEQUENCE:
        if t < duration:
            return v, w
        t -= duration

    return 0.0, 0.0


# =====================================================================
# MAIN
# =====================================================================

def main():

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 200.0,
        device=args_cli.device,
    )
    with launch_simulation(sim_cfg, args_cli):
        run(sim_cfg)


def run(sim_cfg):

    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.1))

    robot = design_scene()

    sim.reset()

    print("[INFO] TraQuad joints:", robot.joint_names)
    print("[INFO] Fixed base:", robot.is_fixed_base)

    left_ids, _ = robot.find_joints(f"{WHEEL_JOINTS}LEFT_.*")
    right_ids, _ = robot.find_joints(f"{WHEEL_JOINTS}RIGHT_.*")
    print(f"[INFO] {len(left_ids)} left, {len(right_ids)} right driven wheels")

    # start directly in the stance and hold legs and ankles there
    pose_target = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=pose_target)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pose_target))
    robot.actuators.target_command.set_position_index(value=pose_target)

    leg_ids, _ = robot.find_joints([".*HFE", "body_.*_ankle"])
    track_ctrl = TrackController(robot, sim.get_physics_dt(),
                                 ankles=args_cli.ankle_stiffness <= 0.0 and not args_cli.rigid_legs,
                                 wheel_comp=not args_cli.no_wheel_comp)

    if args_cli.rigid_legs:
        print("[INFO] Rigid legs: stiff HFE and ankle drives")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    leg_dev = 0.0

    lpf_alpha = sim_dt * 2.0 * math.pi * TWIST_LPF_HZ / (1.0 + sim_dt * 2.0 * math.pi * TWIST_LPF_HZ)
    lin_vel = ang_vel = 0.0

    while sim.is_running():

        lin_cmd, ang_cmd = get_command(sim_time)
        lin_vel += lpf_alpha * (lin_cmd - lin_vel)
        ang_vel += lpf_alpha * (ang_cmd - ang_vel)

        wheel_vel = twist_to_wheel_velocities(robot, left_ids, right_ids, lin_vel, ang_vel)
        robot.actuators.target_command.set_position_index(value=pose_target)
        robot.actuators.target_command.set_velocity_index(value=wheel_vel)
        width = TRACK_WIDTH * TRACK_WIDTH_FACTOR
        track_ctrl.apply(lin_vel - ang_vel * width / 2.0, lin_vel + ang_vel * width / 2.0)
        robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)

        # largest deviation of HFE/ankle joints from the stance since the last print
        leg_dev = max(leg_dev, (robot.data.joint_pos.torch[0, leg_ids] - pose_target[0, leg_ids]).abs().max().item())

        if count % 200 == 0:
            # quaternions are (x, y, z, w) in Isaac Lab 3.0, as expected by quat_apply_inverse
            quat = robot.data.root_quat_w.torch
            lin_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w.torch)[0]
            ang_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w.torch)[0]
            leg_dev_print = leg_dev
            leg_dev = 0.0
            print(
                f"t={sim_time:6.2f}s | cmd v={lin_vel:+.2f} w={ang_vel:+.2f} | "
                f"meas v={lin_b[0].item():+.2f} w={ang_b[2].item():+.2f} | "
                f"max leg dev={leg_dev_print:.4f} rad"
            )


if __name__ == "__main__":

    main()
