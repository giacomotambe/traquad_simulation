# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of the TraQuad robot on mixed terrain: flat, rough, low stairs and boxes, slopes up to 40 deg.

The obstacles stay within what the robot can meet in its use: steps and boxes of at most 10 cm, slopes (pyramids and
inverted pyramids) of at most 40 deg (the real robot climbs 35 deg on wooden boards). Each robot gets one ground
friction for the whole training (startup): static 0.6-1.3, dynamic 0.3-1.0 limited to the static one. The roller dry
friction is a property of the robot and stays fixed (0.06 N m, in the asset).
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass, replace

from . import mdp
from .rough_env_cfg import TraQuadRoughEnvCfg

MAX_STEP = 0.10                            # [m], stairs and boxes
MAX_SLOPE = math.tan(math.radians(40.0))   # slope of the pyramids as rise / run (0.84)

MIXED_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    # 10 cm cells (5 cm cells take 4 times the triangles: out of GPU memory with 1024 robots on 8 GB). A 40 deg slope
    # rises 8.4 cm per cell, below the threshold (9 cm) at which the heightfield is turned into vertical walls; stairs
    # and boxes are meshes, so their steps are vertical at any height
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, MAX_STEP),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, MAX_STEP),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.45,
            grid_height_range=(0.02, MAX_STEP),
            platform_width=2.0,
            convert_to_heightfield=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, MAX_SLOPE), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, MAX_SLOPE), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class TraQuadMixedEnvCfg(TraQuadRoughEnvCfg):
    """Velocity tracking of the TraQuad robot on flat, rough, stepped (<= 10 cm) and sloped (<= 40 deg) terrain."""

    def __post_init__(self):
        super().__post_init__()

        # scene
        # set after the parent __post_init__, which turns on the generator curriculum: turn it on here too
        self.scene.terrain.terrain_generator = replace(MIXED_TERRAINS_CFG, curriculum=True)
        # commands: more robots asked to stand still, so that the policy also learns to hold on slopes
        self.commands.base_velocity.rel_standing_envs = 0.1
        # events: one friction per robot for the whole training
        self.events.ground_friction.params = {
            "static_friction_range": (0.6, 1.3),
            "dynamic_friction_range": (0.3, 1.0),
            "num_buckets": 256,
        }
        # rewards
        # climbing and going down slopes and steps needs vertical speed
        self.rewards.lin_vel_z_l2.weight = -0.02
        # holding still when asked (on a slope it must not slide back): wheel speeds of the standing robots
        self.rewards.stand_still_wheels = RewTerm(
            func=mdp.stand_still_joint_vel_l1,
            weight=-2.0e-3,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_2_.*"]),
            },
        )
        # tracks spinning faster than the base moves (low dynamic friction, slopes the robot cannot climb)
        self.rewards.track_slip = RewTerm(
            func=mdp.track_slip_l1,
            weight=-0.1,
            params={
                "wheel_radius": 0.015,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint_wheel_2_{t}" for t in ("LEFT_F", "LEFT_H", "RIGHT_F", "RIGHT_H")],
                    preserve_order=True,
                ),
            },
        )
