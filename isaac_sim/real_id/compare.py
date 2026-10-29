"""Simulated replay (real_replay.py) vs the real run (robot bag + Qualisys).

Metrics on the replay window, per robot of the replay: RMS error of the body velocities (forward vx, yaw rate wz) on
the Qualisys ground truth, mean of the simulated / real yaw rate and forward speed while turning / driving, track
joint speeds and torques against the measured ones, final pose error of the trajectory integrated from the same start.

usage: python compare.py <run.npz> <sim.npz> [--plot out.png] [--robot k]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import Run, lowpass  # noqa: E402

WH = ['LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL']
HFE = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE']


def real_on(run, t):
    """Real signals on the sim time grid."""
    out = {'vx': np.interp(t, run.qt, run.qvx), 'vy': np.interp(t, run.qt, run.qvy), 'wz': np.interp(t, run.qt, run.qwz),
           'x': np.interp(t, run.qt, run.qp[:, 0]), 'y': np.interp(t, run.qt, run.qp[:, 1]),
           'yaw': np.interp(t, run.qt, run.qyaw)}
    tj = run.t('js')
    sl = (tj >= t[0] - 1) & (tj <= t[-1] + 1)
    for key, names, f in (('w_track', WH, 'velocity'), ('tau_track', WH, 'effort'), ('q_hfe', HFE, 'position')):
        x = np.stack([run.joint('js', f, n)[sl] for n in names], 1)
        x = lowpass(np.nan_to_num(x), 500, 5) if key != 'q_hfe' else x
        out[key] = np.stack([np.interp(t, tj[sl], x[:, j]) for j in range(4)], 1)
    tc = run.t('cmd')
    sc = (tc >= t[0] - 1) & (tc <= t[-1] + 1)
    out['w_cmd'] = np.stack([np.interp(t, tc[sc], run.joint('cmd', 'velocity', n)[sc]) for n in WH], 1)
    tt = run.t('tw')
    k = np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 1)
    out['tw'] = run.d['tw'][k][:, [0, 2]]
    return out


def metrics(real, sim, r, t):
    vx_s, wz_s = lowpass(sim['vx'][:, r], 50, 4), lowpass(sim['wz'][:, r], 50, 4)
    turning = np.abs(real['tw'][:, 1]) > 0.2
    driving = np.abs(real['tw'][:, 0]) > 0.1
    moving = turning | driving
    m = {}
    m['vx_rms'] = np.sqrt(np.mean((vx_s - real['vx'])[moving] ** 2))
    m['wz_rms'] = np.sqrt(np.mean((wz_s - real['wz'])[moving] ** 2))
    # gains: least squares slope sim vs real
    m['wz_gain'] = np.sum(wz_s[turning] * real['wz'][turning]) / np.sum(real['wz'][turning] ** 2)
    m['vx_gain'] = np.sum(vx_s[driving] * real['vx'][driving]) / np.sum(real['vx'][driving] ** 2)
    # track speed (the drive loop: deficit) and torque errors
    wt = sim['w_track'][:, r]
    m['wtrack_rms'] = np.sqrt(np.mean((wt - real['w_track'])[moving] ** 2))
    m['deficit_real'] = np.mean(np.abs(real['w_cmd'] - real['w_track'])[moving])
    m['deficit_sim'] = np.mean(np.abs(sim['w_cmd'][:, r] - wt)[moving])
    tau_s = lowpass(sim['tau_track'][:, r], 50, 5)
    m['tau_rms'] = np.sqrt(np.mean((tau_s - real['tau_track'])[moving] ** 2))
    m['tau_abs_real'] = np.mean(np.abs(real['tau_track'])[moving])
    m['tau_abs_sim'] = np.mean(np.abs(tau_s)[moving])
    # pose: both trajectories expressed from the same start pose
    p_s = sim['pos'][:, r, :2] - sim['pos'][0, r, :2]
    yaw_s = np.unwrap(sim['yaw'][:, r]) - sim['yaw'][0, r]
    c, s = np.cos(-sim['yaw'][0, r]), np.sin(-sim['yaw'][0, r])
    p_s = p_s @ np.array([[c, s], [-s, c]])
    p_r = np.stack([real['x'] - real['x'][0], real['y'] - real['y'][0]], 1)
    c, s = np.cos(-real['yaw'][0]), np.sin(-real['yaw'][0])
    p_r = p_r @ np.array([[c, s], [-s, c]])
    m['pos_err_end'] = np.hypot(*(p_s[-1] - p_r[-1]))
    m['yaw_err_end_deg'] = np.degrees(yaw_s[-1] - (real['yaw'][-1] - real['yaw'][0]))
    m.update(window_errors(real, sim, r, t))
    return m, p_s, p_r


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def window_errors(real, sim, r, t, horizon=10.0, stride=5.0, marker=None):
    """Trajectory error over windows: both trajectories in the frame of their own pose at the window start, compared
    at the end (position [m], heading [deg]). The Qualisys marker sits at an unknown constant offset d from the
    simulated base origin (body frame); its effect on the end position is linear in d and is removed by least squares
    over all windows (or fixed with marker=(dx, dy))."""
    dt = t[1] - t[0]
    n, k = int(round(horizon / dt)), int(round(stride / dt))
    yr, ys = real['yaw'], np.unwrap(sim['yaw'][:, r])
    pr = np.stack([real['x'], real['y']], 1)
    ps = sim['pos'][:, r, :2]
    moving = (np.abs(real['tw']) > [0.05, 0.1]).any(1)
    rows, rhs, yaw_err = [], [], []
    for i0 in range(0, len(t) - n, k):
        i1 = i0 + n
        if moving[i0:i1].mean() < 0.3:
            continue
        er = rot(-yr[i0]) @ (pr[i1] - pr[i0])          # real marker displacement in the start frame
        es = rot(-ys[i0]) @ (ps[i1] - ps[i0])          # simulated base displacement
        dyr, dys = yr[i1] - yr[i0], ys[i1] - ys[i0]
        # marker displacement of the simulated robot: es + (R(dys) - I) d; of the real one: er (marker itself)
        rows.append(rot(dys) - np.eye(2))
        rhs.append(er - es)
        yaw_err.append(np.degrees(dys - dyr))
    if not rows:
        return {}
    A, B = np.concatenate(rows), np.concatenate(rhs)
    d = np.linalg.lstsq(A, B, rcond=None)[0] if marker is None else np.asarray(marker, float)
    res = (B - A @ d).reshape(-1, 2)
    e = np.hypot(res[:, 0], res[:, 1])
    return {'win_pos': float(np.mean(e)), 'win_pos_p90': float(np.percentile(e, 90)),
            'win_yaw': float(np.mean(np.abs(yaw_err))), 'marker_dx': float(d[0]), 'marker_dy': float(d[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('sim')
    ap.add_argument('--plot', default=None)
    ap.add_argument('--robot', type=int, default=0)
    a = ap.parse_args()
    run = Run(Path(a.run).stem, Path(a.run).parent)
    sim = dict(np.load(a.sim))
    t = sim['t']
    real = real_on(run, t)
    names = list(sim['param_names'])
    res = []
    for r in range(sim['vx'].shape[1]):
        m, p_s, p_r = metrics(real, sim, r, t)
        res.append(m)
        par = ' '.join(f'{n}={v:g}' for n, v in zip(names, sim['params'][r]) if np.isfinite(v))
        print(f'robot {r:2d} [{par}]\n   ' + ' '.join(f'{k} {v:.3f}' for k, v in m.items()), flush=True)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        r = a.robot
        m, p_s, p_r = metrics(real, sim, r, t)
        f, ax = plt.subplots(6, 1, figsize=(16, 22), sharex=True)
        ax[0].plot(t, real['tw'][:, 0], 'k:', label='cmd'); ax[0].plot(t, real['vx'], label='real (Qualisys)')
        ax[0].plot(t, lowpass(sim['vx'][:, r], 50, 4), label='sim'); ax[0].set_ylabel('vx [m/s]')
        ax[1].plot(t, real['tw'][:, 1], 'k:'); ax[1].plot(t, real['wz']); ax[1].plot(t, lowpass(sim['wz'][:, r], 50, 4))
        ax[1].set_ylabel('wz [rad/s]')
        ax[2].plot(t, real['vy']); ax[2].plot(t, lowpass(sim['vy'][:, r], 50, 4)); ax[2].set_ylabel('vy [m/s]')
        for j in range(4):
            l, = ax[3].plot(t, real['w_track'][:, j], lw=1, label=WH[j])
            ax[3].plot(t, sim['w_track'][:, r, j], '--', color=l.get_color(), lw=1)
            ax[3].plot(t, real['w_cmd'][:, j], ':', color=l.get_color(), lw=0.8)
            ax[4].plot(t, real['tau_track'][:, j], lw=1, color=l.get_color())
            ax[4].plot(t, lowpass(sim['tau_track'][:, r, j], 50, 5), '--', color=l.get_color(), lw=1)
            ax[5].plot(t, real['q_hfe'][:, j], lw=1, color=l.get_color())
            ax[5].plot(t, sim['q_hfe'][:, r, j], '--', color=l.get_color(), lw=1)
        ax[3].set_ylabel('track speed [rad/s] (real, sim --, cmd :)'); ax[3].legend(ncol=4)
        ax[4].set_ylabel('track torque [N m] (real, sim --)'); ax[5].set_ylabel('HFE [rad]')
        ax[0].legend()
        for x in ax:
            x.grid()
        plt.tight_layout()
        plt.savefig(a.plot, dpi=55)
        f, ax = plt.subplots(figsize=(8, 8))
        ax.plot(*p_r.T, label='real'); ax.plot(*p_s.T, label='sim'); ax.axis('equal'); ax.legend(); ax.grid()
        plt.savefig(a.plot.replace('.png', '_xy.png'), dpi=60)


if __name__ == '__main__':
    main()
