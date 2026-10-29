"""Steady segments of a replay: mean yaw rate and forward speed, real vs simulated, per maneuver.

A segment is a stretch of at least --min_len seconds with a constant joystick twist (after 1 s of settling).
usage: python segments.py <run.npz> <sim.npz> [--robots 0,3] [--min_len 1.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import real_on  # noqa: E402
from runs import Run, lowpass  # noqa: E402


def segments(t, tw, min_len):
    out, i = [], 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and np.all(np.abs(tw[j + 1] - tw[i]) < 0.03):
            j += 1
        if t[j] - t[i] >= min_len and np.abs(tw[i]).max() > 0.05:
            out.append((t[i] + 1.0, t[j], tw[i]))
        i = j + 1
    return out


def table(run_path, sim_path, robots=None, min_len=1.5, hfe_mean=True):
    run = Run(Path(run_path).stem, Path(run_path).parent)
    sim = dict(np.load(sim_path))
    t = sim['t']
    real = real_on(run, t)
    robots = range(sim['vx'].shape[1]) if robots is None else robots
    rows = []
    for a, b, (v, w) in segments(t, real['tw'], min_len):
        m = (t >= a) & (t <= b)
        delta = np.mean(real['q_hfe'][m].sum(1) / 4)
        row = {'t': (a, b), 'v': v, 'w': w, 'delta': delta, 'wz_real': real['wz'][m].mean(), 'vx_real': real['vx'][m].mean()}
        for r in robots:
            row[f'wz_{r}'] = sim['wz'][m, r].mean()
            row[f'vx_{r}'] = sim['vx'][m, r].mean()
        rows.append(row)
    return rows, list(robots), sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('sim')
    ap.add_argument('--robots', default=None)
    ap.add_argument('--min_len', type=float, default=1.5)
    a = ap.parse_args()
    robots = None if a.robots is None else [int(x) for x in a.robots.split(',')]
    rows, robots, sim = table(a.run, a.sim, robots, a.min_len)
    names = list(sim['param_names'])
    for r in robots:
        print(f'robot {r}: ' + ' '.join(f'{n}={v:g}' for n, v in zip(names, sim['params'][r]) if np.isfinite(v)))
    print(f"{'t':>13s} {'v cmd':>6s} {'w cmd':>6s} {'delta':>6s} | {'wz real':>7s} " + ' '.join(f'{"wz r" + str(r):>7s}' for r in robots)
          + f" | {'vx real':>7s} " + ' '.join(f'{"vx r" + str(r):>7s}' for r in robots))
    for row in rows:
        print(f"{row['t'][0]:6.1f}-{row['t'][1]:6.1f} {row['v']:+6.2f} {row['w']:+6.2f} {row['delta']:+6.2f} | "
              f"{row['wz_real']:+7.3f} " + ' '.join(f"{row[f'wz_{r}']:+7.3f}" for r in robots)
              + f" | {row['vx_real']:+7.3f} " + ' '.join(f"{row[f'vx_{r}']:+7.3f}" for r in robots))


if __name__ == '__main__':
    main()
