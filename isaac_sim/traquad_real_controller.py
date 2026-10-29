"""Replica of the real traquad controllers, for the simulators (numpy only, no Isaac imports).

Given the twist commanded by the joystick (vx, wz) it gives the same track joint velocities and hip references as
the robot:

- ``YawPosture``: R2 mode of the NUC joystick (mulinex_app omni_mulinex_joystick, ``yaw_posture_step`` /
  ``yaw_posture_band``): common offset delta of the four HFE joints from the commanded yaw rate, two first-order
  stages and a rate limit, 100 Hz; arc band (inner track below v_min) with the SBC IK model.
- ``OmniController``: SBC omni_controller (mulsbc_ws) with ``feet_type: traquad``, 500 Hz: twist clamp, 2 Hz first
  order low-pass, ``TraquadIK`` (effective half track y_eff and forward creep from the measured shoulder angles, inner
  track kept moving in arcs, optional track deficit compensation) and the per-wheel ``wheel_compensation``
  (friction feedforward + integral of the wheel's own speed error, sent as effort).
- ``moteus_torque``: the drives, torque = Kp (q_ref - q) + Kd (dq_ref - dq) + effort, joint side.

All in joint coordinates of the real robot: track joint speed [rad/s] (belt speed = WHEEL_RAD x joint speed),
HFE angles [rad]. The simulators convert the track joint to their own wheels (``track_joint_to_sim``).
Joint order everywhere: LF, LH, RF, RH.
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

LEGS = ('LF', 'LH', 'RF', 'RH')
SIDE = np.array([1, 1, -1, -1])        # kappa: +1 left, -1 right (forward belt = kappa x joint speed)
SIGMA = np.array([1, -1, -1, 1])       # alpha = sigma q_HFE (alpha > 0 = leg down)
WHEEL_RAD = 0.045                      # m of belt per rad of the track joint (Qualisys 2026-10-22)

# Drives (mulsbc_ws traquad8_dji.xacro): joint-side gains
HFE_KP, HFE_KD = 60.0, 0.6
TRACK_KD = 0.9                         # Kp 0: velocity loop
TRACK_MAX_VELOCITY = 25.0              # rad/s, servo.max_velocity of the track drives
MAX_FEEDFORWARD = 15.0                 # N m, max_effort: clamp of the effort command only


@dataclass
class IKConfig:
    """traquad8_dji.yaml, omni_controller (HEAD of mulsbc_ws, 2026-10-23)."""
    wheel_rad: float = WHEEL_RAD
    track_width: float = 0.4
    shoulder_x: float = 0.235
    leg_length: float = 0.19
    yaw_slip_gain: float = 1.35        # gamma
    yaw_posture_gain: float = 0.29     # k_delta [m/rad]
    creep_gain: float = 1.6            # [1/rad]
    creep_x0: float = 0.171            # [m]
    creep_posture: float = 0.09        # [m/rad^2]
    min_track_speed: float = 1.5       # [rad/s] 0 = off
    min_track_hysteresis: float = 0.5  # [rad/s]
    deficit_compensation: bool = False
    deficit_base: float = 1.215
    deficit_speed: float = 0.025
    deficit_turn: float = -0.221
    deficit_side: float = 0.939
    deficit_fade: float = 1.0


@dataclass
class WheelCompConfig:
    enabled: bool = True
    friction: float = 1.4              # N m
    viscous: float = 0.017             # N m s/rad
    fade: float = 1.0                  # rad/s
    ki: float = 3.0                    # N m/rad
    i_max: float = 1.0                 # N m


@dataclass
class ControllerConfig:
    ik: IKConfig = field(default_factory=IKConfig)
    wheel_comp: WheelCompConfig = field(default_factory=WheelCompConfig)
    lpf_cutoff: float = 2.0            # Hz, velocity_lpf
    twist_max_x: float = 1.0
    twist_max_z: float = 1.0           # default of the omni_controller (not in the yaml)
    stand: float = 1.6                 # q_stand shoulder angle


def config(preset='current'):
    """Controller settings of the real runs.

    current: HEAD of mulsbc_ws / mulinex_app (2026-10-23 16:46).
    The 2026-10-23 runs were recorded before the last commit: no min_track_speed (and no arc band in the joystick),
    and their own compensation:
      sim_ident     (missions_r2_sim_ident, teleop 721-881 s): no compensation
      deficit       (ik_deficit_compensation): track deficit compensation in the IK
      lowlevel      (lowlevel_wheel_compensation): per-wheel compensation in the controller
    """
    c = ControllerConfig()
    if preset == 'current':
        return c
    c.ik.min_track_speed = 0.0
    if preset == 'sim_ident':
        c.wheel_comp.enabled = False
    elif preset == 'deficit':
        c.wheel_comp.enabled = False
        c.ik.deficit_compensation = True
    elif preset == 'lowlevel':
        pass
    else:
        raise ValueError(preset)
    return c


class TraquadIK:
    """mulsbc_ws omni_controller/src/wheel_ik.cpp, TraquadIK."""

    def __init__(self, cfg: IKConfig):
        self.c = cfg
        self.h = cfg.track_width / 2
        self.inner_sign = 0
        self.set_leg_state(np.zeros(4))

    def set_leg_state(self, q_hfe):
        c = self.c
        x_mean = np.mean(c.shoulder_x + c.leg_length * np.cos(SIGMA * q_hfe))
        delta = abs(np.sum(q_hfe)) / 4
        ratio = x_mean / self.h
        self.y_eff = max(self.h * (1 + c.yaw_slip_gain * ratio ** 2) - c.yaw_posture_gain * delta, self.h)
        self.creep = max(c.creep_gain * (x_mean - c.creep_x0) - c.creep_posture * delta, 0.0)

    def _deficit(self, w, load, outer):
        c = self.c
        d = c.deficit_base + c.deficit_speed * abs(w) + load * (c.deficit_turn + (1 if outer else -1) * c.deficit_side)
        return max(d, 0.0) * min(abs(w) / c.deficit_fade, 1.0)

    def _keep_inner_track_moving(self, vx, wz, vl, vr):
        c = self.c
        v_min = c.min_track_speed * c.wheel_rad
        if abs(vx) < v_min or abs(wz) < 1e-3:
            self.inner_sign = 0
            return vl, vr
        left_inner = abs(vl) < abs(vr)
        inner, outer = (vl, vr) if left_inner else (vr, vl)
        if abs(inner) >= v_min:
            self.inner_sign = 0
            return vl, vr
        h = c.min_track_hysteresis * c.wheel_rad
        if self.inner_sign == 0:
            self.inner_sign = (1 if inner > 0 else -1) if abs(inner) > h else (1 if vx > 0 else -1)
        elif inner > h:
            self.inner_sign = 1
        elif inner < -h:
            self.inner_sign = -1
        target = self.inner_sign * v_min
        outer += target - inner
        inner = target
        return (inner, outer) if left_inner else (outer, inner)

    def inverse(self, vx, wz):
        """Track joint speeds LF, LH, RF, RH [rad/s]."""
        c = self.c
        v = vx - self.creep * abs(wz)
        vl, vr = v - self.y_eff * wz, v + self.y_eff * wz
        if c.min_track_speed > 0:
            vl, vr = self._keep_inner_track_moving(vx, wz, vl, vr)
        s, dlt = abs(vl + vr), abs(vr - vl)
        load = 1 - s / (s + dlt) if s + dlt > 1e-9 else 0.0
        out = np.zeros(4)
        for i, side in enumerate(SIDE):
            own, other = (vl, vr) if side > 0 else (vr, vl)
            w = own / c.wheel_rad
            if c.deficit_compensation:
                outer = (own - other) * own > 0
                w += math.copysign(self._deficit(w, load, outer), w)
            out[i] = side * w
        return out


class OmniController:
    """SBC omni_controller, ACTIVE state, wheel_mode ik, feet_type traquad. Call ``step`` at 500 Hz."""

    def __init__(self, cfg: ControllerConfig = None, dt=0.002):
        self.c = cfg or config()
        self.dt = dt
        self.ik = TraquadIK(self.c.ik)
        self.twist = np.zeros(2)       # vx, wz after the clamp
        self.filtered = None
        self.integral = np.zeros(4)
        self.wheel_cmd = np.zeros(4)
        self.wheel_effort = np.zeros(4)

    def set_twist(self, vx, wz):
        """~/twist_cmd callback (the joystick publishes at 10 Hz)."""
        self.twist = np.array([np.clip(vx, -self.c.twist_max_x, self.c.twist_max_x),
                               np.clip(wz, -self.c.twist_max_z, self.c.twist_max_z)])

    def step(self, q_hfe, dq_wheel):
        """One control cycle: measured HFE angles and track joint speeds -> track speed and effort commands."""
        if self.filtered is None:
            self.filtered = self.twist.copy()
        else:
            wc = 2 * math.pi * self.c.lpf_cutoff
            a = min(max(self.dt * wc / (1 + self.dt * wc), 0.0), 1.0)
            self.filtered = a * self.twist + (1 - a) * self.filtered
        self.ik.set_leg_state(np.asarray(q_hfe, float))
        self.wheel_cmd = self.ik.inverse(*self.filtered)
        wc = self.c.wheel_comp
        if wc.enabled:
            ff = np.sign(self.wheel_cmd) * (wc.friction + wc.viscous * np.abs(self.wheel_cmd)) * \
                np.minimum(np.abs(self.wheel_cmd) / wc.fade, 1.0)
            self.integral = np.clip(self.integral + wc.ki * (self.wheel_cmd - dq_wheel) * self.dt, -wc.i_max, wc.i_max)
            self.wheel_effort = ff + self.integral
        else:
            self.wheel_effort = np.zeros(4)
        return self.wheel_cmd, self.wheel_effort


@dataclass
class PostureConfig:
    """omni_mulinex_joystick yaw_posture.* (HEAD of mulinex_app)."""
    delta_max: float = 0.30
    omega0: float = 0.5
    v0: float = 1.0
    tau: float = 0.15
    max_rate: float = 1.0
    stand: float = 1.6
    alpha_min: float = 1.1
    alpha_max: float = 2.1
    dt: float = 0.01
    band: bool = True
    band_v_min: float = 1.5
    band_delta_max: float = 0.45
    band_hysteresis: float = 0.3
    band_dwell: float = 0.3
    band_wz_max: float = 1.0


class YawPosture:
    """R2 mode of the joystick: hip references from the commanded twist. Call ``step`` at 100 Hz."""

    def __init__(self, cfg: PostureConfig = None, ik: IKConfig = None):
        self.c = cfg or PostureConfig()
        self.k = ik or IKConfig()
        self.active = False
        self.delta = self.filter = 0.0
        self.rate = 0.0
        self.band_active, self.band_side, self.band_in, self.band_out = False, 0, 0.0, 0.0

    def _band(self, target, vx, wz):
        c, k = self.c, self.k

        def reset():
            self.band_active, self.band_side, self.band_in, self.band_out = False, 0, 0.0, 0.0
        if not c.band:
            reset()
            return target
        v_min = c.band_v_min * k.wheel_rad
        wz = float(np.clip(wz, -c.band_wz_max, c.band_wz_max))
        if abs(vx) < v_min or abs(wz) < 0.05:
            reset()
            return target
        h = k.track_width / 2
        c_stand = math.cos(c.stand)

        def gap(d):
            xbar = k.shoulder_x + k.leg_length * c_stand * math.cos(d)
            y = h * (1 + k.yaw_slip_gain * (xbar / h) ** 2) - k.yaw_posture_gain * d
            creep = max(k.creep_gain * (xbar - k.creep_x0) - k.creep_posture * d, 0.0)
            return abs(vx - creep * abs(wz)) / abs(wz) - y
        m = v_min / abs(wz)
        d_normal = abs(target)
        d_max = max(c.band_delta_max, d_normal)
        g_normal = gap(d_normal)
        self.band_in = self.band_in + c.dt if abs(g_normal) < m else 0.0
        self.band_out = self.band_out + c.dt if abs(g_normal) > m * (1 + c.band_hysteresis) else 0.0
        if self.band_active and self.band_out >= c.band_dwell:
            reset()
        margin = 1.1 * m

        def stronger():
            if gap(d_max) < margin:
                return -1.0
            lo, hi = d_normal, d_max
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if gap(mid) >= margin:
                    hi = mid
                else:
                    lo = mid
            return hi

        def weaker():
            if gap(0.0) > -margin:
                return -1.0
            lo, hi = 0.0, d_normal
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if gap(mid) <= -margin:
                    lo = mid
                else:
                    hi = mid
            return lo
        if not self.band_active and self.band_in >= c.band_dwell:
            up, down = stronger(), weaker()
            if up >= 0 or down >= 0:
                self.band_side = 1 if (down < 0 or (up >= 0 and up - d_normal <= d_normal - down)) else -1
                self.band_active = True
        if not self.band_active:
            return target
        d = stronger() if self.band_side > 0 else weaker()
        if d < 0:
            d = weaker() if self.band_side > 0 else stronger()
        if d < 0:
            return target
        return math.copysign(d, wz)

    def step(self, vx, wz):
        """vx, wz: twist of the sticks (before the SBC clamp). Returns HFE position and velocity references."""
        c = self.c
        target = self._band(c.delta_max * math.tanh(wz / c.omega0) / (1 + abs(vx) / c.v0), vx, wz) \
            if self.active else 0.0
        tau = max(c.tau, c.dt)
        self.filter += c.dt / tau * (target - self.filter)
        self.rate = float(np.clip((self.filter - self.delta) / tau, -c.max_rate, c.max_rate))
        self.delta += self.rate * c.dt
        return self.references()

    def references(self):
        c = self.c
        alpha = c.stand + SIGMA * self.delta
        alpha_c = np.clip(alpha, c.alpha_min, c.alpha_max)
        q = SIGMA * alpha_c
        dq = np.where(alpha_c == alpha, self.rate, 0.0)
        return q, dq


def moteus_torque(q, dq, q_ref, dq_ref, effort, kp, kd, kp_scale=1.0, kd_scale=1.0, max_velocity=None):
    """Joint torque of a moteus drive (position NaN -> kp_scale 0). Velocity command limited to max_velocity."""
    if max_velocity is not None:
        dq_ref = np.clip(dq_ref, -max_velocity, max_velocity)
    return kp * kp_scale * (q_ref - q) + kd * kd_scale * (dq_ref - dq) + np.clip(effort, -MAX_FEEDFORWARD, MAX_FEEDFORWARD)


class RealRobotController:
    """Joystick (with or without R2) + omni_controller + drives: twist in, joint torques / targets out.

    ``step(t, q_hfe, dq_hfe, dq_track)`` at the control rate (500 Hz); the posture runs every 10 ms, the twist is
    read through ``set_twist`` (call it at 10 Hz to mimic the joystick, or every step).
    """

    def __init__(self, preset='current', r2=False, dt=0.002):
        self.cfg = config(preset)
        self.ctrl = OmniController(self.cfg, dt)
        self.posture = YawPosture(PostureConfig(stand=self.cfg.stand,
                                                band=preset == 'current'), self.cfg.ik)
        self.posture.active = r2
        self.dt = dt
        self.t_posture = 0.0
        self.stick = np.zeros(2)
        self.q_ref, self.dq_ref = self.posture.references()

    def set_twist(self, vx, wz):
        self.stick = np.array([vx, wz])
        self.ctrl.set_twist(vx, wz)

    def set_r2(self, on):
        self.posture.active = on

    def step(self, q_hfe, dq_hfe, dq_track):
        self.t_posture += self.dt
        if self.t_posture >= self.posture.c.dt - 1e-9:
            self.t_posture -= self.posture.c.dt
            self.q_ref, self.dq_ref = self.posture.step(*self.stick)
        w_cmd, w_eff = self.ctrl.step(q_hfe, dq_track)
        tau_hfe = moteus_torque(q_hfe, dq_hfe, self.q_ref, self.dq_ref, 0.0, HFE_KP, HFE_KD)
        tau_track = moteus_torque(0.0, dq_track, 0.0, w_cmd, w_eff, 0.0, TRACK_KD, max_velocity=TRACK_MAX_VELOCITY)
        return tau_hfe, tau_track
