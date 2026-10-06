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

from isaaclab.managers import SceneEntityCfg

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


def feet_air_time_stuck_recovery(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    stuck_velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward steps longer than ``threshold`` [s], only while the robot is stuck.

    The reward is active only when the planar speed of the base is below ``stuck_velocity_threshold`` [m/s],
    to encourage the robot to lift the bodies of ``sensor_cfg`` and unstick itself.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # only when the robot is stuck
    robot_speed = torch.linalg.norm(asset.data.root_lin_vel_w.torch[:, :2], dim=1)
    return reward * (robot_speed < stuck_velocity_threshold)


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
