"""Score the robots of one or more replays (same parameter grid) against the real runs (sorted by the trajectory error
over 10 s windows, compare.window_errors).

Per robot and window: RMS error of the yaw rate and of the forward speed while moving (compare.metrics), and the
effective half track y_eff = (v_R - v_L) / (2 wz) from the track speeds of each steady turning segment (real
Qualisys and simulated), whose relative error does not depend on how fast the tracks actually ran.
usage: python score.py <run.npz>:<sim.npz> [<run.npz>:<sim.npz> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import metrics, real_on  # noqa: E402
from runs import Run  # noqa: E402
from segments import segments  # noqa: E402

RS = 0.045


def y_eff(w_track, wz):
    vl = w_track[:, :2].mean(1) * RS
    vr = -w_track[:, 2:].mean(1) * RS
    return ((vr - vl) / 2).mean() / wz.mean()


def score(pairs):
    out = {}
    for run_path, sim_path in pairs:
        run = Run(Path(run_path).stem, Path(run_path).parent)
        sim = dict(np.load(sim_path))
        t = sim['t']
        real = real_on(run, t)
        segs = [(a, b) for a, b, (v, w) in segments(t, real['tw'], 1.0) if abs(w) >= 0.2]
        for r in range(sim['vx'].shape[1]):
            key = tuple(np.round(np.nan_to_num(sim['params'][r], nan=-1.0), 6))
            m, _, _ = metrics(real, sim, r, t)
            ye = []
            for a, b in segs:
                k = (t >= a) & (t <= b)
                yr, ys = y_eff(real['w_track'][k], real['wz'][k]), y_eff(sim['w_track'][k, r], sim['wz'][k, r])
                ye.append(ys / yr - 1)
            d = out.setdefault(key, {'names': list(sim['param_names'])})
            d[Path(sim_path).stem] = {'wz_rms': m['wz_rms'], 'vx_rms': m['vx_rms'], 'yeff_err': np.mean(ye),
                                      'win_pos': m.get('win_pos', np.nan), 'win_yaw': m.get('win_yaw', np.nan),
                                      'marker': (m.get('marker_dx', np.nan), m.get('marker_dy', np.nan)),
                                      'yeff_abs': np.sqrt(np.mean(np.square(ye)))}
    return out


if __name__ == '__main__':
    pairs = [a.split(':') for a in sys.argv[1:]]
    out = score(pairs)
    rows = []
    for key, d in out.items():
        wins = [k for k in d if k != 'names']
        # trajectory score: mean end error of the 10 s windows, position [m] + heading [deg] / 20
        tot = np.mean([d[w]['win_pos'] + d[w]['win_yaw'] / 20 for w in wins])
        par = ' '.join(f'{n}={v:g}' for n, v in zip(d['names'], key) if v != -1.0 and n not in (
            'mu', 'roller_damping', 'track_friction', 'track_viscous'))
        rows.append((tot, par, ' | '.join(f"{w}: win {d[w]['win_pos']:.2f} m {d[w]['win_yaw']:.0f} deg d ({d[w]['marker'][0]:+.2f},{d[w]['marker'][1]:+.2f}) y_eff {d[w]['yeff_err']:+.0%} wz_rms {d[w]['wz_rms']:.3f}"
                                         for w in wins)))
    for tot, par, txt in sorted(rows):
        print(f'{tot:.3f}  [{par}]  {txt}')
