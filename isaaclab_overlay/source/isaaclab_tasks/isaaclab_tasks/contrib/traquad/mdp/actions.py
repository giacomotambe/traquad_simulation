# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms of the TraQuad tasks."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.utils import configclass


class JointVelocityActionGroup(JointVelocityAction):
    """Joint velocity action with a single action for the whole group of joints.

    The same velocity command is applied to all the joints of the group (e.g. the 4 wheels of a track), so the
    policy outputs one value per group.
    """

    cfg: JointVelocityActionGroupCfg

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions, shape (num_envs, 1)
        self._raw_actions[:] = actions
        # repeat the action on all the joints of the group and apply the affine transformation
        self._processed_actions = actions.repeat(1, self._num_joints) * self._scale + self._offset


@configclass
class JointVelocityActionGroupCfg(JointVelocityActionCfg):
    """Configuration for :class:`JointVelocityActionGroup`.

    Only a float :attr:`scale` is supported, and :attr:`clip` is ignored.
    """

    class_type: type[JointVelocityActionGroup] = JointVelocityActionGroup
