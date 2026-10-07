# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the TraQuad tasks.

The velocity command is (vx, vy, wz) with vy always zero: the robot is skid-steered and does not move sideways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the forward velocity command (x axis of the base) using an exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b.torch[:, 0]
    )
    return torch.exp(-lin_vel_x_error / std**2)


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the lateral (y axis) base velocity with an L2 squared kernel: the command never asks for it."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power of the joints of ``asset_cfg``, sum of |torque * velocity| [W].

    Unlike the squared torque it costs nothing to hold a pose, only to move: on the legs it makes the robot keep
    them still (drive like a differential-drive robot) unless moving them pays off in tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * vel), dim=1)


class track_air_time_stuck_recovery(ManagerTermBase):
    """Reward lifting a whole track for longer than ``threshold`` [s] while the robot is stuck.

    A track is in contact when any of its bodies (track body, wheels, rollers) touches something: single rollers
    touch and leave the ground every few milliseconds while the wheel turns, so the contact is taken per track.
    The robot is stuck when the command asks for a forward speed above ``moving_command_threshold`` [m/s] and the
    planar speed of the base is below ``stuck_velocity_threshold`` [m/s]. Only the time spent in the air while stuck
    counts; when the track lands, the term returns that time minus ``threshold``.
    """

    TRACKS = (("left", "F", "LEFT_F"), ("left", "H", "LEFT_H"), ("right", "F", "RIGHT_F"), ("right", "H", "RIGHT_H"))

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_name"]]
        self.track_ids = [
            self.sensor.find_sensors(f"body_{s}_{fh}|wheel_\\d_{name}.*")[0] for s, fh, name in self.TRACKS
        ]
        self.air_time = torch.zeros(env.num_envs, len(self.TRACKS), device=env.device)
        self.in_contact = torch.ones(env.num_envs, len(self.TRACKS), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.air_time[env_ids] = 0.0
        self.in_contact[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_name: str,
        threshold: float,
        force_threshold: float = 1.0,
        command_name: str = "base_velocity",
        moving_command_threshold: float = 0.1,
        stuck_velocity_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        force = torch.linalg.norm(self.sensor.data.net_forces_w.torch, dim=-1)
        contact = torch.stack([(force[:, ids] > force_threshold).any(dim=1) for ids in self.track_ids], dim=1)
        asset: Articulation = env.scene[asset_cfg.name]
        speed = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
        command = env.command_manager.get_command(command_name)
        stuck = (torch.abs(command[:, 0]) > moving_command_threshold) & (speed < stuck_velocity_threshold)
        landed = contact & ~self.in_contact & (self.air_time > 0.0)
        reward = torch.sum((self.air_time - threshold) * landed, dim=1)
        self.air_time = torch.where(contact, 0.0, self.air_time + env.step_dt * stuck.unsqueeze(1))
        self.in_contact = contact
        return reward


def joint_vel_sign_disagreement(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """1 when the first two joints of ``asset_cfg`` turn in opposite directions, 0 otherwise.

    Use it on two wheels of the same track (``preserve_order=True``), one term per side.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return (vel[:, 0] * vel[:, 1] < 0).float()


def joint_vel_difference(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute difference between the velocities of the first two joints of ``asset_cfg`` [rad/s].

    Use it on two wheels of the same track (``preserve_order=True``), one term per side.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.abs(vel[:, 0] - vel[:, 1])


def stand_still_joint_vel_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint speeds [rad/s] of the robots whose command is zero (L1), zero for the others."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.norm(command[:, :3], dim=1) < 1e-3
    return torch.sum(torch.abs(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1) * standing


def track_slip_l1(env: ManagerBasedRLEnv, wheel_radius: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Longitudinal slip of the tracks [m/s]: mean belt speed minus forward speed of the base.

    ``asset_cfg`` takes one wheel per track in the order LEFT_F, LEFT_H, RIGHT_F, RIGHT_H. The wheel joint axes are
    along -y on the left and +y on the right, so the belt speed is -r w on the left and r w on the right. The mean over
    the four tracks removes the opposite slips of left and right in turns, which skid steering needs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    belt = wheel_radius * (-w[:, 0] - w[:, 1] + w[:, 2] + w[:, 3]) / 4.0
    return torch.abs(belt - asset.data.root_lin_vel_b.torch[:, 0])
