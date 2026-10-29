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
    drives the ankle of the track when the track is in the air (drive sprocket coaxial with the ankle), with the
    controller of the track: the implicit ankle drive gets the damping :attr:`speed_gain` (the track Kd) and the target
    wheel target x :attr:`ankle_velocity_ratio`, and the integral of the speed error (:attr:`integral_gain`, limited to
    :attr:`max_torque`) is added as effort, so the frame turns with the sprocket until an end stop. When any body of
    the track touches something (contact sensor) the drive goes back to :attr:`passive_damping` with target 0, the
    effort is the sprocket torque on the track (:attr:`sprocket_torque_ratio` x the drive torque of the track wheels)
    and the integral is reset. The ankle actuator must be implicit.

    The wheels get the wheel compensation of the real track controller (mulsbc_ws omni_controller, wheel_compensation)
    as effort on top of their implicit velocity drive (the moteus Kd): friction feedforward
    sign(w) (:attr:`comp_friction` + :attr:`comp_viscous` |w|), faded below :attr:`comp_fade`, plus the integral of the
    track speed error (:attr:`comp_ki`, limited to :attr:`comp_i_max`), all at the track joint (sprocket) and referred
    to the wheels with :attr:`track_ratio` (wheel radius / sprocket radius) and split over the wheels of the group.
    """

    cfg: JointVelocityActionGroupCfg

    def __init__(self, cfg: JointVelocityActionGroupCfg, env):
        super().__init__(cfg, env)
        self._ankle_ids = None
        self._dt = env.physics_dt
        self._comp_integral = torch.zeros(self.num_envs, 1, device=self.device)
        if cfg.ankle_joint_name is not None:
            ids, _ = self._asset.find_joints(cfg.ankle_joint_name)
            self._ankle_ids = torch.as_tensor(ids, device=self.device)
            self._sensor = env.scene.sensors[cfg.contact_sensor_name]
            self._track_bodies = self._sensor.find_sensors(cfg.track_body_expr)[0]
            if not self._track_bodies:
                raise ValueError(f"No contact sensor body matches {cfg.track_body_expr}")
            self._integral = torch.zeros(self.num_envs, 1, device=self.device)
            self._air_time = torch.zeros(self.num_envs, 1, device=self.device)
            self._was_in_air = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

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
        self._apply_wheel_compensation()
        if self._ankle_ids is None:
            return
        # called on every physics step: the ankle follows the sprocket with the controller of the track while in the air
        force = torch.linalg.norm(self._sensor.data.net_forces_w.torch[:, self._track_bodies], dim=-1)
        touching = (force > self.cfg.contact_force_threshold).any(dim=1, keepdim=True)
        # debounce: the controller takes over only after the track has been in the air for air_delay seconds
        self._air_time = torch.where(touching, 0.0, self._air_time + self._dt)
        in_air = self._air_time > self.cfg.air_delay
        target = self._processed_actions[:, :1] * self.cfg.ankle_velocity_ratio
        err = target - self._asset.data.joint_vel.torch[:, self._ankle_ids]
        self._integral = torch.where(
            in_air, (self._integral + self.cfg.integral_gain * err * self._dt).clamp(-self.cfg.max_torque,
                                                                                     self.cfg.max_torque), 0.0
        )
        if (in_air != self._was_in_air).any():
            damping = torch.where(in_air, self.cfg.speed_gain, self.cfg.passive_damping)
            self._asset.write_joint_damping_to_sim_index(damping=damping, joint_ids=self._ankle_ids)
            self._was_in_air = in_air.clone()
        self._asset.set_joint_velocity_target_index(target=torch.where(in_air, target, 0.0), joint_ids=self._ankle_ids)
        wheel_torque = self._asset.actuators.applied_effort.torch[:, self._joint_ids].sum(dim=1, keepdim=True)
        sprocket = self.cfg.sprocket_torque_ratio * wheel_torque
        self._asset.set_joint_effort_target_index(target=torch.where(in_air, self._integral, sprocket),
                                                  joint_ids=self._ankle_ids)

    def _apply_wheel_compensation(self):
        # track joint speed and command (sprocket), in the spin direction of the wheels
        cfg = self.cfg
        w_cmd = self._processed_actions[:, :1] * cfg.track_ratio
        w = self._asset.data.joint_vel.torch[:, self._joint_ids].mean(dim=1, keepdim=True) * cfg.track_ratio
        ff = torch.sign(w_cmd) * (cfg.comp_friction + cfg.comp_viscous * w_cmd.abs()) * (w_cmd.abs() / cfg.comp_fade).clamp(
            max=1.0)
        self._comp_integral = (self._comp_integral + cfg.comp_ki * (w_cmd - w) * self._dt).clamp(-cfg.comp_i_max,
                                                                                                 cfg.comp_i_max)
        tau_wheel = (ff + self._comp_integral) * cfg.track_ratio / self._num_joints
        self._asset.set_joint_effort_target_index(target=tau_wheel.expand(-1, self._num_joints),
                                                  joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._comp_integral[ids] = 0.0
        if self._ankle_ids is not None:
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

    speed_gain: float = 0.9
    """Damping of the implicit ankle drive while the track is in the air [N m s/rad] (track Kd)."""

    integral_gain: float = 3.0
    """Integral gain of the ankle speed error while the track is in the air [N m/rad] (wheel compensation ki)."""

    max_torque: float = 1.0
    """Limit of the integral term [N m] (wheel compensation i_max)."""

    passive_damping: float = 0.01
    """Damping of the ankle drive while the track touches something [N m s/rad]."""

    sprocket_torque_ratio: float = 0.0
    """Ankle torque per unit of summed wheel drive torque of the track, applied on the ground (0: none)."""

    air_delay: float = 0.05
    """Time the track must spend without contact before the ankle controller takes over [s]."""

    track_ratio: float = 1.0
    """Track joint (sprocket) speed per unit of wheel speed, wheel radius / sprocket radius."""

    comp_friction: float = 0.0
    """Friction feedforward of the wheel compensation at the track joint [N m] (0: none)."""

    comp_viscous: float = 0.0
    """Viscous feedforward of the wheel compensation at the track joint [N m s/rad]."""

    comp_fade: float = 1.0
    """Track speed command below which the feedforward fades linearly to zero [rad/s]."""

    comp_ki: float = 0.0
    """Integral gain of the wheel compensation on the track speed error [N m/rad] (0: none)."""

    comp_i_max: float = 0.0
    """Limit of the integral term of the wheel compensation [N m]."""
