"""Ankle model of the traquad for the Isaac Sim scripts (same law as the Isaac Lab tasks).

On the real robot the track motor drives a sprocket coaxial with the ankle and the track frame pivots on the same
axis between end stops. Model:
- passive damping of the pivot, PASSIVE_DAMPING [N m s/rad], always (a velocity drive with zero target);
- track touching the ground: nothing else, the ankle is passive;
- track in the air: the frame turns with the sprocket at the speed that gives the commanded belt speed,
  w_target = v_track / SPROCKET_RADIUS, held by a PI velocity controller (the integral removes the steady error due
  to gravity, so the track keeps turning at w_target, however small, until an end stop):
  tau = clip(SPEED_GAIN * e + INTEGRAL_GAIN * int(e dt), +-MAX_TORQUE),  e = w_target - dq_ankle.
  The integral is reset when the track touches the ground and frozen while the torque is saturated (anti-windup).
  The controller takes over only after AIR_DELAY seconds without contact (debounce).
The controller runs on every physics step (pre-step callback). Contact is taken from the height of the wheels above a
flat ground at z = 0 (wheel centre lower than the wheel radius plus CONTACT_MARGIN), so these scripts need a flat
ground.
"""
import numpy as np

WHEEL_RADIUS = 0.015
SPROCKET_RADIUS = 0.015      # placeholder until measured on the robot
PASSIVE_DAMPING = 0.01       # [N m s/rad]
SPEED_GAIN = 0.2             # [N m s/rad]
INTEGRAL_GAIN = 2.0          # [N m/rad]
MAX_TORQUE = 0.05            # [N m]
CONTACT_MARGIN = 0.004       # [m]
AIR_DELAY = 0.05             # [s], time without contact before the controller takes over
TRACKS = ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')


class AnkleController:
    """Velocity control of the ankles when the tracks are in the air.

    Set the passive damping with ``kd[ankles] = PASSIVE_DAMPING`` before ``set_dof_gains``; give the belt speed of the
    left and right tracks with ``set_speeds``. The controller then runs by itself on every physics step.
    """

    def __init__(self, robot, enabled=True):
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self.robot, self.enabled = robot, enabled
        names = robot.dof_names
        links = robot.link_names
        self.ankle = [names.index(f"body_{t.split('_')[0].lower()}_{t.split('_')[1]}_ankle") for t in TRACKS]
        self.wheels = [[links.index(f'wheel_{k}_{t}') for k in (1, 2, 3, 4)] for t in TRACKS]
        self.left = np.array([t.startswith('LEFT') for t in TRACKS])
        self.speeds = (0.0, 0.0)
        self.integral = np.zeros(len(TRACKS))
        self.in_air = np.zeros(len(TRACKS), bool)
        self.air_time = np.zeros(len(TRACKS))
        self.last_tau = np.zeros(len(TRACKS), np.float32)
        if enabled:
            self._cb = SimulationManager.register_callback(self._step, event=SimulationEvent.PHYSICS_PRE_STEP)

    def set_speeds(self, v_left, v_right):
        self.speeds = (v_left, v_right)

    def _step(self, dt, context=None):
        tf = self.robot._physics_articulation_view.get_link_transforms().numpy()[0]
        dq = self.robot.get_dof_velocities().numpy()[0][self.ankle]
        off_ground = np.array([tf[w, 2].min() > WHEEL_RADIUS + CONTACT_MARGIN for w in self.wheels])
        self.air_time = np.where(off_ground, self.air_time + dt, 0.0)
        self.in_air = self.air_time > AIR_DELAY   # debounce
        # ankle axes along +y on both sides: the sprocket (and the hanging frame) turns with the wheels
        target = np.where(self.left, self.speeds[0], self.speeds[1]) / SPROCKET_RADIUS
        err = target - dq
        tau_free = SPEED_GAIN * err + INTEGRAL_GAIN * self.integral
        saturated = np.abs(tau_free) >= MAX_TORQUE
        # anti-windup: integrate only while the output is not saturated (or the error brings it back)
        grow = ~saturated | (np.sign(err) != np.sign(tau_free))
        self.integral = np.where(self.in_air, self.integral + np.where(grow, err * dt, 0.0), 0.0)
        tau = np.where(self.in_air, np.clip(SPEED_GAIN * err + INTEGRAL_GAIN * self.integral, -MAX_TORQUE, MAX_TORQUE), 0.0)
        self.last_tau = tau.astype(np.float32)
        self.robot.set_dof_efforts(self.last_tau[None], dof_indices=np.array(self.ankle))
