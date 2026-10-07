"""Ankle model of the traquad for the Isaac Sim scripts (same law as the Isaac Lab tasks).

On the real robot the track motor sits on the leg and drives a sprocket coaxial with the ankle; the track frame pivots
on the same axis between end stops. Model:
- passive damping of the pivot, PASSIVE_DAMPING [N m s/rad], always (a velocity drive with zero target);
- track touching the ground: the ankle is passive, plus the torque the sprocket puts on the track. In the simulation the
  motor drives the track wheels, so its reaction stays in the track frame and the traction F pitches the track with
  F * h (h = height of the pivot above the ground). On the robot the motor drives the belt from the sprocket on the
  pivot: the belt receives the sprocket torque F * r_s from the leg and the pitch moment is F * (h - r_s). The ankle gets
  that torque, (r_s / r_wheel) * sum of the wheel drive torques of the track (in the spin direction of the wheels);
- track in the air: the frame turns with the sprocket at the speed that gives the commanded belt speed,
  w_target = v_track / r_s, held by a PI velocity controller (the integral removes the steady error due to gravity, so
  the track keeps turning at w_target, however small, until an end stop):
  tau = clip(SPEED_GAIN * e + INTEGRAL_GAIN * int(e dt), +-MAX_TORQUE),  e = w_target - dq_ankle.
  The integral is reset when the track touches the ground and frozen while the torque is saturated (anti-windup).
  The controller takes over only after AIR_DELAY seconds without contact (debounce).
The controller runs on every physics step (pre-step callback). Contact is taken from the height of the wheels above a
flat ground at z = 0 (wheel centre lower than the wheel radius plus CONTACT_MARGIN), so these scripts need a flat
ground (a tilted gravity is fine).
"""
import numpy as np

WHEEL_RADIUS = 0.015
SPROCKET_RADIUS = 0.025      # placeholder until measured: smallest value that climbs 35 deg at mu 0.75 without tipping
PASSIVE_DAMPING = 0.01       # [N m s/rad]
SPEED_GAIN = 0.2             # [N m s/rad]
INTEGRAL_GAIN = 2.0          # [N m/rad]
MAX_TORQUE = 0.05            # [N m]
CONTACT_MARGIN = 0.004       # [m]
AIR_DELAY = 0.05             # [s], time without contact before the controller takes over
TRACKS = ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')


class AnkleController:
    """Ankle model: sprocket torque on the ground, velocity control of the ankle in the air.

    Set the passive damping with ``kd[ankles] = PASSIVE_DAMPING`` before ``set_dof_gains``; give the belt speed of the
    left and right tracks with ``set_speeds``. The controller then runs by itself on every physics step. ``wheel_kd`` and
    ``wheel_max`` (per driven wheel) enable the sprocket torque on the ground; ``followers`` are mimic wheels (no drive).
    """

    def __init__(self, robot, enabled=True, wheel_kd=None, wheel_max=None, followers=(), sprocket_radius=None):
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self.robot, self.enabled = robot, enabled
        self.r_s = SPROCKET_RADIUS if sprocket_radius is None else sprocket_radius
        names = robot.dof_names
        links = robot.link_names
        self.ankle = [names.index(f"body_{t.split('_')[0].lower()}_{t.split('_')[1]}_ankle") for t in TRACKS]
        self.wheels = [[links.index(f'wheel_{k}_{t}') for k in (1, 2, 3, 4)] for t in TRACKS]
        self.driven = [[names.index(f'joint_wheel_{k}_{t}') for k in (1, 2, 3, 4)
                        if f'joint_wheel_{k}_{t}' not in followers] for t in TRACKS]
        self.left = np.array([t.startswith('LEFT') for t in TRACKS])
        # wheel joint axes along -y (left) and +y (right), ankle axes along +y: spin sign of each side in ankle terms
        self.side = np.where(self.left, -1.0, 1.0)
        self.wheel_kd, self.wheel_max = wheel_kd, wheel_max
        self.speeds = (0.0, 0.0)
        self.integral = np.zeros(len(TRACKS))
        self.in_air = np.zeros(len(TRACKS), bool)
        self.air_time = np.zeros(len(TRACKS))
        self.last_tau = np.zeros(len(TRACKS), np.float32)
        self.last_sprocket = np.zeros(len(TRACKS))
        if enabled:
            self._cb = SimulationManager.register_callback(self._step, event=SimulationEvent.PHYSICS_PRE_STEP)

    def set_speeds(self, v_left, v_right):
        self.speeds = (v_left, v_right)

    def _step(self, dt, context=None):
        tf = self.robot._physics_articulation_view.get_link_transforms().numpy()[0]
        dq_all = self.robot.get_dof_velocities().numpy()[0]
        dq = dq_all[self.ankle]
        off_ground = np.array([tf[w, 2].min() > WHEEL_RADIUS + CONTACT_MARGIN for w in self.wheels])
        self.air_time = np.where(off_ground, self.air_time + dt, 0.0)
        self.in_air = self.air_time > AIR_DELAY   # debounce
        v = np.where(self.left, self.speeds[0], self.speeds[1])
        # in the air: PI velocity control towards the sprocket speed
        target = v / self.r_s
        err = target - dq
        tau_free = SPEED_GAIN * err + INTEGRAL_GAIN * self.integral
        saturated = np.abs(tau_free) >= MAX_TORQUE
        # anti-windup: integrate only while the output is not saturated (or the error brings it back)
        grow = ~saturated | (np.sign(err) != np.sign(tau_free))
        self.integral = np.where(self.in_air, self.integral + np.where(grow, err * dt, 0.0), 0.0)
        tau_air = np.clip(SPEED_GAIN * err + INTEGRAL_GAIN * self.integral, -MAX_TORQUE, MAX_TORQUE)
        # on the ground: torque of the sprocket on the track (drive torque entering through the pivot)
        sprocket = np.zeros(len(TRACKS))
        if self.wheel_kd is not None:
            for j, ids in enumerate(self.driven):
                w_target = self.side[j] * v[j] / WHEEL_RADIUS
                tw = np.clip(self.wheel_kd * (w_target - dq_all[ids]), -self.wheel_max, self.wheel_max).sum()
                sprocket[j] = self.side[j] * tw * self.r_s / WHEEL_RADIUS   # joint torques -> ankle (+y) terms
        self.last_sprocket = sprocket
        tau = np.where(self.in_air, tau_air, sprocket)
        self.last_tau = tau.astype(np.float32)
        self.robot.set_dof_efforts(self.last_tau[None], dof_indices=np.array(self.ankle))
