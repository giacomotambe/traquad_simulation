# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the TraQuad tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def base_lin_acc_b(
    env: ManagerBasedEnv, clip: float = 30.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link")
) -> torch.Tensor:
    """Linear acceleration of the base in the base frame [m/s^2], gravity removed (the accelerometer of the robot,
    whose gravity offset is subtracted on board), clipped to +-``clip`` (contact spikes of the solver)."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_com_lin_acc_w.torch[:, asset_cfg.body_ids[0]]
    acc_b = quat_apply_inverse(asset.data.root_quat_w.torch, acc_w)
    return acc_b.clamp(-clip, clip)
