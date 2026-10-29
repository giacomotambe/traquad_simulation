"""Spin / straight / arc segments of the identification missions (t1-t3): real vs simulated, per shoulder angle.

For each segment (markers /omni_mulinex_joystick/identification/segment, 1 s settling left out): effective half track
y_eff = (v_R - v_L) / (2 wz) from the measured track speeds and forward creep per unit yaw rate d = vx / |wz|.
usage: python missions.py <run.npz> <sim.npz> [--kind spin]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import real_on  # noqa: E402
from runs import Run  # noqa: E402

RS = 0.045


def seg_list(run, kind):
    d = run.d
    ts = d['seg_t'] - run.t0
    segs = [json.loads(s) for s in d['seg_json']]
    out = []
    for i, s in enumerate(segs[:-1]):
        if s['segment'] == kind:
            out.append((ts[i] + 1.0, ts[i + 1], s))
    return out


def track_stats(w_track, wz, vx):
    vl = w_track[:, :2].mean(1) * RS
    vr = -w_track[:, 2:].mean(1) * RS
    w = wz.mean()
    return ((vr - vl) / 2).mean() / w, vx.mean() / abs(w), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('sim')
    ap.add_argument('--kind', default='spin')
    a = ap.parse_args()
    run = Run(Path(a.run).stem, Path(a.run).parent)
    sim = dict(np.load(a.sim))
    t = sim['t']
    real = real_on(run, t)
    names = list(sim['param_names'])
    nr = sim['vx'].shape[1]
    for r in range(nr):
        print(f'robot {r}: ' + ' '.join(f'{n}={v:g}' for n, v in zip(names, sim['params'][r])
                                        if np.isfinite(v) and n in ('roller_friction', 'hfe_offset', 'contact_stiffness',
                                                                     'mu_dyn', 'sprocket')))
    by_alpha = {}
    for a0, b0, s in seg_list(run, a.kind):
        m = (t >= a0) & (t <= b0)
        if m.sum() < 20:
            continue
        al = round(s['alpha_F'], 2)
        yr, dr, wr = track_stats(real['w_track'][m], real['wz'][m], real['vx'][m])
        row = [yr, dr, wr]
        for r in range(nr):
            row += list(track_stats(sim['w_track'][m, r], sim['wz'][m, r], sim['vx'][m, r]))
        by_alpha.setdefault(al, []).append(row)
    print(f"{'alpha':>5s} {'n':>2s} | {'y_eff real':>10s} " + ' '.join(f'{"r" + str(r):>6s}' for r in range(nr))
          + f" | {'d real':>7s} " + ' '.join(f'{"r" + str(r):>6s}' for r in range(nr)))
    for al, rows in sorted(by_alpha.items()):
        x = np.median(np.array(rows), 0)
        print(f'{al:5.2f} {len(rows):2d} | {x[0]:10.3f} ' + ' '.join(f'{x[3 + 3 * r]:6.3f}' for r in range(nr))
              + f' | {x[1]:+7.3f} ' + ' '.join(f'{x[4 + 3 * r]:+6.3f}' for r in range(nr)))


if __name__ == '__main__':
    main()
