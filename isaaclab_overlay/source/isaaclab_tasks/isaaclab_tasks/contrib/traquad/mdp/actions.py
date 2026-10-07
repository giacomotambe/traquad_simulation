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
    policy outputs one value per group. With :attr:`JointVelocityActionGroupCfg.ankle_joint_name` the same command also
    drives the ankle of the track when the track is in the air (drive sprocket coaxial with the ankle): a PI velocity
    controller, run on every physics step, turns the ankle at wheel target x
    :attr:`JointVelocityActionGroupCfg.ankle_velocity_ratio`, so the track keeps turning at that speed, however small,
    until an end stop. When any body of the track touches something (contact sensor) the ankle gets instead the torque
    of the sprocket on the track, :attr:`JointVelocityActionGroupCfg.sprocket_torque_ratio` x the drive torque of the
    track wheels (the simulated motor drives the wheels, the real one drives the belt from the pivot), and the integral
    is reset.
    """

    cfg: JointVelocityActionGroupCfg

    def __init__(self, cfg: JointVelocityActionGroupCfg, env):
        super().__init__(cfg, env)
        self._ankle_ids = None
        if cfg.ankle_joint_name is not None:
            ids, _ = self._asset.find_joints(cfg.ankle_joint_name)
            self._ankle_ids = torch.as_tensor(ids, device=self.device)
            self._sensor = env.scene.sensors[cfg.contact_sensor_name]
            self._track_bodies = self._sensor.find_sensors(cfg.track_body_expr)[0]
            if not self._track_bodies:
                raise ValueError(f"No contact sensor body matches {cfg.track_body_expr}")
            self._integral = torch.zeros(self.num_envs, 1, device=self.device)
            self._air_time = torch.zeros(self.num_envs, 1, device=self.device)
            self._dt = env.physics_dt

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions, shape (num_envs, 1)
        self._raw_actions[:] = actions
        # repeat the action on all the joints of the group and apply the affine transformation
        self._processed_actions = actions.repeat(1, self._num_joints) * self._scale + self._offset

    def apply_actions(self):
        super().apply_actions()
        if self._ankle_ids is None:
            return
        # called on every physics step: PI velocity control of the ankle while the track is in the air
        force = torch.linalg.norm(self._sensor.data.net_forces_w.torch[:, self._track_bodies], dim=-1)
        touching = (force > self.cfg.contact_force_threshold).any(dim=1, keepdim=True)
        # debounce: the controller takes over only after the track has been in the air for air_delay seconds
        self._air_time = torch.where(touching, 0.0, self._air_time + self._dt)
        in_air = self._air_time > self.cfg.air_delay
        err = self._processed_actions[:, :1] * self.cfg.ankle_velocity_ratio - self._asset.data.joint_vel.torch[
            :, self._ankle_ids
        ]
        tau_free = self.cfg.speed_gain * err + self.cfg.integral_gain * self._integral
        # anti-windup: integrate only while the output is not saturated (or the error brings it back)
        grow = (tau_free.abs() < self.cfg.max_torque) | (torch.sign(err) != torch.sign(tau_free))
        self._integral = torch.where(in_air, self._integral + torch.where(grow, err * self._dt, 0.0), 0.0)
        tau = self.cfg.speed_gain * err + self.cfg.integral_gain * self._integral
        wheel_torque = self._asset.actuators.applied_effort.torch[:, self._joint_ids].sum(dim=1, keepdim=True)
        sprocket = self.cfg.sprocket_torque_ratio * wheel_torque
        tau = torch.where(in_air, tau.clamp(-self.cfg.max_torque, self.cfg.max_torque), sprocket)
        self._asset.set_joint_effort_target_index(target=tau, joint_ids=self._ankle_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self._ankle_ids is not None:
            ids = slice(None) if env_ids is None else env_ids
            self._integral[ids] = 0.0
            self._air_time[ids] = 0.0


@configclass
class JointVelocityActionGroupCfg(JointVelocityActionCfg):
    """Configuration for :class:`JointVelocityActionGroup`.

    Only a float :attr:`scale` is supported, and :attr:`clip` is ignored.
    """

    class_type: type[JointVelocityActionGroup] = JointVelocityActionGroup

    ankle_joint_name: str | None = None
    """Ankle of the track driven by this group, velocity controlled when the track is in the air (None: no control)."""

    ankle_velocity_ratio: float = 0.0
    """Ankle velocity target per unit of wheel velocity target (in the air)."""

    contact_sensor_name: str = "contact_forces"
    """Contact sensor of the scene that covers the bodies of the track."""

    track_body_expr: str = ""
    """Regular expression of the bodies of the track (frame, wheels, rollers) in the contact sensor."""

    contact_force_threshold: float = 1.0
    """Contact force above which a body of the track is touching something [N]."""

    speed_gain: float = 0.2
    """Proportional gain of the ankle velocity controller [N m s/rad]."""

    integral_gain: float = 2.0
    """Integral gain of the ankle velocity controller [N m/rad]."""

    max_torque: float = 0.05
    """Torque limit of the ankle velocity controller [N m]."""

    sprocket_torque_ratio: float = 0.0
    """Ankle torque per unit of summed wheel drive torque of the track, applied on the ground (0: none)."""

    air_delay: float = 0.05
    """Time the track must spend without contact before the ankle controller takes over [s]."""
