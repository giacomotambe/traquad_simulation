"""Load a converted run, align the Qualisys clock to the robot one and give the base motion in the body frame."""
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

WORK = Path(__file__).resolve().parent


def quat_yaw(q):  # xyzw
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lowpass(x, fs, fc, axis=0):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=axis)


class Run:
    def __init__(self, name, folder=WORK):
        self.name = name
        d = dict(np.load(Path(folder) / f'{name}.npz'))
        self.d = d
        self.t0 = d['js_t'][0]
        self.names = list(d['js_names'])
        self._align()

    def t(self, key):
        return self.d[key + '_t'] - self.t0

    def _align(self):
        d = self.d
        # Qualisys on a uniform 200 Hz grid of its own clock, gaps filled linearly
        tq = d['qtm_t']
        p, q = d['qtm_p'], d['qtm_q']
        ok = np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)
        tq, p, q = tq[ok], p[ok], q[ok]
        yaw = np.unwrap(quat_yaw(q))
        g = np.arange(tq[0], tq[-1], 0.005)
        self.q_gap = np.interp(g, tq[1:], np.diff(tq)) if len(tq) > 1 else None
        P = np.stack([np.interp(g, tq, p[:, i]) for i in range(3)], 1)
        Y = np.interp(g, tq, yaw)
        wq = np.gradient(lowpass(Y, 200, 8), g)
        # IMU yaw rate on the same grid, robot clock
        ti = d['imu_t']
        wi = d['imu_w'][:, 2]
        gi = np.arange(ti[0], ti[-1], 0.005)
        Wi = lowpass(np.interp(gi, ti, wi), 200, 8)
        # cross-correlation: offset such that robot_time = qtm_time + off
        n = len(gi)
        best = None
        f = np.fft.rfft(Wi - Wi.mean(), 2 * max(n, len(g)))
        h = np.fft.rfft(wq - wq.mean(), 2 * max(n, len(g)))
        cc = np.fft.irfft(f * np.conj(h))
        k = int(np.argmax(cc))
        N = len(cc)
        lag = k if k < N // 2 else k - N
        off = gi[0] - g[0] + lag * 0.005
        # refine by least squares on a fine grid
        best = (np.inf, off)
        for o in np.arange(off - 0.05, off + 0.05, 0.001):
            r = np.interp(gi, g + o, wq, left=np.nan, right=np.nan) - Wi
            e = np.nanmean(r ** 2)
            if e < best[0]:
                best = (e, o)
        self.qtm_offset = best[1]
        self.align_rms = np.sqrt(best[0])
        self.qt = g + self.qtm_offset - self.t0
        self.qp = P
        self.qyaw = Y
        # body-frame velocities (Qualisys rigid-body origin) from smoothed positions
        Ps = lowpass(P, 200, 4)
        Ys = lowpass(Y, 200, 4)
        v = np.gradient(Ps, g, axis=0)
        c, s = np.cos(Ys), np.sin(Ys)
        self.qvx = c * v[:, 0] + s * v[:, 1]
        self.qvy = -s * v[:, 0] + c * v[:, 1]
        self.qwz = np.gradient(Ys, g)

    def joint(self, key, field, name):
        names = list(self.d[key + '_names'])
        return self.d[f'{key}_{field}'][:, names.index(name)]

    def at(self, t, tt, x):
        return np.interp(t, tt, x)


if __name__ == '__main__':
    import sys
    for n in sys.argv[1:]:
        r = Run(n)
        print(n, 'robot', r.d['js_t'][-1] - r.t0, 's  qtm offset', r.qtm_offset, 'rms', r.align_rms,
              'qtm covers', r.qt[0], r.qt[-1], 'names', r.names, list(r.d['ref_names']))
