# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of the TraQuad robot on rough terrain.

The robot is skid-steered: the command is a forward velocity and a yaw rate (lateral velocity always zero).
The policy controls the 4 HFE joints in position and each track (4 wheels) with a single velocity action.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.traquad import TRAQUAD_CFG  # isort: skip

# the links of the TraQuad USD are nested under Robot/Geometry
BASE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"

# track velocity action scale [rad/s of the wheels per unit of action]: 1 unit = 1.3 m/s of track speed (r = 15 mm).
# With |v| <= 1 m/s and |w| <= 1 rad/s a track needs up to v + w * B / 2 = 1.2 m/s, plus the skid-steering slip
WHEEL_VEL_SCALE = 1.3 / 0.015


##
# Scene definition
##


@configclass
class TraQuadSceneCfg(MySceneCfg):
    """Rough terrain scene with the TraQuad robot."""

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*", history_length=3, track_air_time=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        heading_command=False,
        debug_vis=True,
        # skid steering: no lateral velocity
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
    )


def _track_wheels(side: str) -> list[str]:
    return [f"joint_wheel_{i}_{side}" for i in range(1, 5)]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*HFE"], scale=0.5, use_default_offset=True
    )
    # one velocity action per track
    joint_vel_LF = mdp.JointVelocityActionGroupCfg(
        asset_name="robot", joint_names=_track_wheels("LEFT_F"), scale=WHEEL_VEL_SCALE
    )
    joint_vel_LH = mdp.JointVelocityActionGroupCfg(
        asset_name="robot", joint_names=_track_wheels("LEFT_H"), scale=WHEEL_VEL_SCALE
    )
    joint_vel_RF = mdp.JointVelocityActionGroupCfg(
        asset_name="robot", joint_names=_track_wheels("RIGHT_F"), scale=WHEEL_VEL_SCALE
    )
    joint_vel_RH = mdp.JointVelocityActionGroupCfg(
        asset_name="robot", joint_names=_track_wheels("RIGHT_H"), scale=WHEEL_VEL_SCALE
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (no base linear velocity)."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_HFE = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        joint_vel_wheel_2 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_2_.*"])},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for the critic: the policy ones plus base linear velocity and height scan, no noise."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    # the terrain: one wheel-ground friction per robot (slippery floor to rubber on asphalt), dynamic <= static.
    # With the roller dry friction of the asset (0.06 N m, fixed: a property of the robot) the rollers stay locked
    # when parked, so the lateral grip of the robot follows this friction (slides at tan(slope) > mu)
    ground_friction = EventTerm(
        func=mdp.randomize_ground_friction,
        mode="startup",
        params={"static_friction_range": (0.3, 1.0), "dynamic_ratio_range": (0.8, 1.0), "num_buckets": 256},
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.1, 0.3),
            "torque_range": (-0.2, 0.2),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.15)}
    )
    # yaw tracking is the hard part of skid steering: same weight as the forward velocity
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    # the command never asks for lateral speed: lets the policy compensate the outward drift in turns
    lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.1)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])}
    )
    # mechanical power of the legs: holding a pose is free, moving the legs costs. The robot drives like a
    # differential-drive robot on easy ground and moves the legs only when that pays off (steps, rough terrain).
    # Kept light at the start so the policy still explores leg motions; raise it later for more energy saving
    hfe_power_l1 = RewTerm(
        func=mdp.joint_power_l1, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])}
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])}
    )
    # raw actions: with exploration noise 1.0 the sum over the 8 actions is about 16, keep it light at the start
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HFE"])}
    )
    # the ankles are passive: the policy acts on them only indirectly
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["body_.*_ankle"])},
    )
    # stuck (forward speed commanded, base not moving): reward lifting a whole track, e.g. onto a step
    track_air_time_stuck = RewTerm(
        func=mdp.track_air_time_stuck_recovery,
        weight=0.25,
        params={"sensor_name": "contact_forces", "threshold": 0.05},
    )
    # -- the two middle wheels (wheel 2) of the front and hind track of a side should turn together
    wheel_disagreement_L = RewTerm(
        func=mdp.joint_vel_sign_disagreement,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_wheel_2_LEFT_F", "joint_wheel_2_LEFT_H"], preserve_order=True
            )
        },
    )
    wheel_disagreement_R = RewTerm(
        func=mdp.joint_vel_sign_disagreement,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_wheel_2_RIGHT_F", "joint_wheel_2_RIGHT_H"], preserve_order=True
            )
        },
    )
    # [rad/s]: weight scaled with the wheel action scale (10 -> 86.7 rad/s per unit), as -0.005 was with 10
    wheel_vel_difference_L = RewTerm(
        func=mdp.joint_vel_difference,
        weight=-5.8e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_wheel_2_LEFT_F", "joint_wheel_2_LEFT_H"], preserve_order=True
            )
        },
    )
    wheel_vel_difference_R = RewTerm(
        func=mdp.joint_vel_difference,
        weight=-5.8e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_wheel_2_RIGHT_F", "joint_wheel_2_RIGHT_H"], preserve_order=True
            )
        },
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 0.1},
    )


##
# Environment configuration
##


@configclass
class TraQuadRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Velocity tracking of the TraQuad robot on rough terrain."""

    # 121 bodies per robot (rollers included): 1024 envs take about 4.5 GB of GPU memory
    scene: TraQuadSceneCfg = TraQuadSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = replace(TRAQUAD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = BASE_PRIM_PATH
        self.scene.terrain.max_init_terrain_level = 1
        # the roller model has 121 links: with more than 32 instances and more than 64 links, GPU articulations with
        # several partitions corrupt the articulation state (robots launched, NaN): isaac-sim/IsaacLab#8121
        self.sim.physics.isaacsim_physx.gpu_max_num_partitions = 1
        # the roller dry friction of the USD is a PhysX joint attribute
        self.sim.physics.default = self.sim.physics.isaacsim_physx
