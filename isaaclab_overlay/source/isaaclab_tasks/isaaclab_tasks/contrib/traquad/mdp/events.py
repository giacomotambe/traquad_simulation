# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the TraQuad tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class randomize_ground_friction(ManagerTermBase):
    """Give each robot one wheel-ground friction, the same on all its colliders (PhysX).

    It stands for the terrain the robot drives on: static friction sampled in ``static_friction_range``, dynamic
    friction = static friction x a factor in ``dynamic_ratio_range`` (<= 1, so dynamic <= static). The ground of the
    velocity tasks has friction 1.0 with the "multiply" combine mode, which wins over the robot's "average", so the
    contact friction is the sampled value.

    The generic ``randomize_rigid_body_material`` samples a material per collider: on the TraQuad (about 115
    colliders) each roller gets a different friction, the robot ends up with an average value and sees neither
    slippery nor high-grip ground. Values come from ``num_buckets`` pre-sampled materials, to stay far from the PhysX
    limit of 64000 unique materials.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        n = int(cfg.params["num_buckets"])
        s_lo, s_hi = cfg.params["static_friction_range"]
        r_lo, r_hi = cfg.params["dynamic_ratio_range"]
        static = torch.empty(n).uniform_(s_lo, s_hi)
        dynamic = static * torch.empty(n).uniform_(r_lo, r_hi)
        self.buckets = torch.stack([static, dynamic, torch.zeros(n)], dim=1)   # (static, dynamic, restitution)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_ratio_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)
        materials = wp.to_torch(self.asset.root_view.get_material_properties())
        bucket = torch.randint(0, len(self.buckets), (len(env_ids),))
        materials[env_ids.long()] = self.buckets[bucket].unsqueeze(1).expand(-1, materials.shape[1], -1)
        self.asset.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )
