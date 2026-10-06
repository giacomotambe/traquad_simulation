# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the TraQuad tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def randomize_roller_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_roller_.*"]),
):
    """Set the dry (Coulomb) friction torque of the roller joints to a value sampled uniformly per robot [N m].

    Static and dynamic friction get the same value, the viscous friction keeps its value. The generic
    ``randomize_joint_parameters`` cannot be used here: it applies the same range to the viscous friction too,
    which would lock the rollers.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
    tau = torch.empty(len(env_ids), 1, device=asset.device).uniform_(*friction_range).expand(-1, len(joint_ids))
    viscous = asset.data.joint_viscous_friction_coeff.torch[env_ids][:, joint_ids]
    asset.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=tau.contiguous(),
        joint_dynamic_friction_coeff=tau.contiguous(),
        joint_viscous_friction_coeff=viscous.contiguous(),
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
