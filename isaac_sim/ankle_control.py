"""Ankle model of the traquad for the Isaac Sim scripts (same law as the Isaac Lab tasks).

On the real robot the track motor sits on the leg and drives a sprocket coaxial with the ankle; the track frame pivots
on the same axis between end stops. Model:
- passive damping of the pivot, PASSIVE_DAMPING [N m s/rad], always (a velocity drive with zero target);
- track touching the ground: the ankle is passive, plus the torque the sprocket puts on the track. In the simulation the
  motor drives the track wheels, so its reaction stays in the track frame and the traction F pitches the track with
  F * h (h = height of the pivot above the ground). On the robot the motor drives the belt from the sprocket on the
  pivot: the belt receives the sprocket torque F * r_s from the leg and the pitch moment is F * (h - r_s). The ankle gets
  that torque, (r_s / r_wheel) * sum of the wheel drive torques of the track (in the spin direction of the wheels);
- track in the air: the frame turns with the sprocket, driven by the controller of its track with the same speed
  command, w_target = v_track / r_s: the track Kd (TRACK_KD, moteus, as PhysX velocity drive of the ankle, implicit) and
  the integral of the wheel compensation of the omni_controller (TRACK_KI, limited to +-TRACK_I_MAX) as effort; the
  integral is reset when the track touches the ground (mulsbc_ws traquad8_dji.xacro / .yaml). Until 2026-10-27 this
  was a weak PI (Kp 0.2, Ki 2, max 0.05 N m). The controller takes over only after AIR_DELAY seconds without contact
  (debounce).
The wheels get the wheel compensation of the real track controller (omni_controller wheel_compensation, enabled on the
robot): friction feedforward sign(w) (WHEEL_COMP_FRICTION + WHEEL_COMP_VISCOUS |w|), faded below WHEEL_COMP_FADE, plus
the integral of the track speed error (TRACK_KI, +-TRACK_I_MAX), at the track joint and referred to the driven wheels, as
effort on top of their velocity drive (the moteus Kd).
The controller runs on every physics step (pre-step callback). Contact is taken from the height of the wheels above a
flat ground at z = 0 (wheel centre lower than the wheel radius plus CONTACT_MARGIN), so these scripts need a flat
ground (a tilted gravity is fine).
"""
import numpy as np

WHEEL_RADIUS = 0.015
SPROCKET_RADIUS = 0.045      # wheel_rad of the real robot IK (mulsbc_ws traquad8_dji.yaml, Qualisys 2026-10-22; was 0.028)
SPROCKET_TORQUE_FACTOR = 0.0  # scale of the sprocket torque on the ground (1 = the model above). Off since 2026-10-26: with
                              # r_s 45 mm it tips the tracks onto their stops on flat ground (track_test.py: 4% yaw, no
                              # straight driving); without it the robot climbs 30 deg at mu 0.6 but not 35 deg at mu 0.75
PASSIVE_DAMPING = 0.01       # [N m s/rad]
TRACK_KD = 0.9               # [N m s/rad], moteus Kd of the track joint
TRACK_KI = 3.0               # [N m/rad], wheel_compensation.ki of the omni_controller
TRACK_I_MAX = 1.0            # [N m], wheel_compensation.i_max
WHEEL_COMP_FRICTION = 1.4    # [N m], wheel_compensation.friction (feedforward)
WHEEL_COMP_VISCOUS = 0.017   # [N m s/rad], wheel_compensation.viscous
WHEEL_COMP_FADE = 1.0        # [rad/s], wheel_compensation.fade
CONTACT_MARGIN = 0.004       # [m]
AIR_DELAY = 0.05             # [s], time without contact before the controller takes over
TRACKS = ('LEFT_F', 'LEFT_H', 'RIGHT_F', 'RIGHT_H')


class AnkleController:
    """Ankle model: sprocket torque on the ground, velocity control of the ankle in the air.

    Set the passive damping with ``kd[ankles] = PASSIVE_DAMPING`` before ``set_dof_gains`` (the controller switches the
    ankle drive to TRACK_KD in the air and back to PASSIVE_DAMPING on the ground); give the belt speed of the
    left and right tracks with ``set_speeds``. The controller then runs by itself on every physics step. ``wheel_kd`` and
    ``wheel_max`` (per driven wheel) enable the sprocket torque on the ground; ``followers`` are mimic wheels (no drive).
    ``enabled`` switches the ankle control, ``wheel_comp`` the wheel compensation on the driven wheels.
    """

    def __init__(self, robot, enabled=True, wheel_kd=None, wheel_max=None, followers=(), sprocket_radius=None,
                 wheel_comp=True):
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self.robot, self.enabled, self.wheel_comp = robot, enabled, wheel_comp
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
        self.air_prev = np.zeros(len(TRACKS), bool)
        self.air_time = np.zeros(len(TRACKS))
        self.last_tau = np.zeros(len(TRACKS), np.float32)
        self.last_sprocket = np.zeros(len(TRACKS))
        self.comp_integral = np.zeros(len(TRACKS))
        self.last_comp = np.zeros(len(TRACKS))
        if enabled or wheel_comp:
            self._cb = SimulationManager.register_callback(self._step, event=SimulationEvent.PHYSICS_PRE_STEP)

    def set_speeds(self, v_left, v_right):
        self.speeds = (v_left, v_right)

    def _step(self, dt, context=None):
        tf = self.robot._physics_articulation_view.get_link_transforms().numpy()[0]
        dq_all = self.robot.get_dof_velocities().numpy()[0]
        v = np.where(self.left, self.speeds[0], self.speeds[1])
        if self.wheel_comp:
            self._wheel_compensation(dq_all, v, dt)
        if not self.enabled:
            return
        dq = dq_all[self.ankle]
        off_ground = np.array([tf[w, 2].min() > WHEEL_RADIUS + CONTACT_MARGIN for w in self.wheels])
        self.air_time = np.where(off_ground, self.air_time + dt, 0.0)
        self.in_air = self.air_time > AIR_DELAY   # debounce
        # in the air: the controller of the track on the ankle, Kd as drive (implicit) and the integral as effort
        target = v / self.r_s
        err = target - dq
        self.integral = np.where(self.in_air, np.clip(self.integral + TRACK_KI * err * dt, -TRACK_I_MAX, TRACK_I_MAX), 0.0)
        tau_air = self.integral
        if (self.in_air != self.air_prev).any():
            kd = np.where(self.in_air, TRACK_KD, PASSIVE_DAMPING).astype(np.float32)
            self.robot.set_dof_gains(stiffnesses=np.zeros((1, 4), np.float32), dampings=kd[None],
                                     dof_indices=np.array(self.ankle))
            self.air_prev = self.in_air.copy()
        self.robot.set_dof_velocity_targets(np.where(self.in_air, target, 0.0).astype(np.float32)[None],
                                            dof_indices=np.array(self.ankle))
        # on the ground: torque of the sprocket on the track (drive torque entering through the pivot)
        sprocket = np.zeros(len(TRACKS))
        if self.wheel_kd is not None:
            for j, ids in enumerate(self.driven):
                w_target = self.side[j] * v[j] / WHEEL_RADIUS
                tw = np.clip(self.wheel_kd * (w_target - dq_all[ids]), -self.wheel_max, self.wheel_max).sum()
                sprocket[j] = self.side[j] * tw * self.r_s / WHEEL_RADIUS   # joint torques -> ankle (+y) terms
        sprocket = SPROCKET_TORQUE_FACTOR * sprocket
        self.last_sprocket = sprocket
        tau = np.where(self.in_air, tau_air, sprocket)
        self.last_tau = tau.astype(np.float32)
        self.robot.set_dof_efforts(self.last_tau[None], dof_indices=np.array(self.ankle))

    def _wheel_compensation(self, dq_all, v, dt):
        # track joint (sprocket) speed and command in the spin direction of the wheels
        ratio = WHEEL_RADIUS / self.r_s
        w_cmd = self.side * v / self.r_s
        w = np.array([dq_all[ids].mean() for ids in self.driven]) * ratio
        ff = np.sign(w_cmd) * (WHEEL_COMP_FRICTION + WHEEL_COMP_VISCOUS * np.abs(w_cmd)) * \
            np.minimum(np.abs(w_cmd) / WHEEL_COMP_FADE, 1.0)
        self.comp_integral = np.clip(self.comp_integral + TRACK_KI * (w_cmd - w) * dt, -TRACK_I_MAX, TRACK_I_MAX)
        self.last_comp = ff + self.comp_integral
        ids = np.concatenate([np.array(d) for d in self.driven])
        tau = np.concatenate([np.full(len(d), self.last_comp[j] * ratio / len(d)) for j, d in enumerate(self.driven)])
        self.robot.set_dof_efforts(tau.astype(np.float32)[None], dof_indices=ids)
