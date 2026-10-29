"""Motion of the real robot for the 'ghost' of real_replay.py --video: base pose from the Qualisys (aligned to the robot
clock, marker offset removed), hip angles and track joint speeds from the encoders.

The Qualisys marker sits at a constant offset (dx, dy) from the base origin in the body frame (compare.window_errors
estimates it: about +0.06, -0.04 m).
usage: python ghost.py <run.npz> <out.npz> [--marker 0.06 -0.04] [--t0 a --t1 b]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import Run, lowpass  # noqa: E402

WH = ['LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL']
HFE = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE']

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('out')
ap.add_argument('--marker', type=float, nargs=2, default=[0.06, -0.04])
ap.add_argument('--t0', type=float, default=None)
ap.add_argument('--t1', type=float, default=None)
a = ap.parse_args()
run = Run(Path(a.run).stem, Path(a.run).parent)
t0 = a.t0 if a.t0 is not None else run.qt[0]
t1 = a.t1 if a.t1 is not None else run.qt[-1]
t = np.arange(t0, t1, 0.01)
yaw = np.interp(t, run.qt, lowpass(run.qyaw, 200, 6))
mx = np.interp(t, run.qt, lowpass(run.qp[:, 0], 200, 6))
my = np.interp(t, run.qt, lowpass(run.qp[:, 1], 200, 6))
c, s = np.cos(yaw), np.sin(yaw)
bx = mx - (c * a.marker[0] - s * a.marker[1])
by = my - (s * a.marker[0] + c * a.marker[1])
tj = run.t('js')
q = np.stack([np.interp(t, tj, run.joint('js', 'position', n)) for n in HFE], 1)
w = np.stack([np.interp(t, tj, lowpass(run.joint('js', 'velocity', n), 500, 10)) for n in WH], 1)
vx = np.interp(t, run.qt, run.qvx)
wz = np.interp(t, run.qt, run.qwz)
# wheel angles of the ghost (sim wheel axes: wheel speed = -track joint speed * 0.045 / 0.015)
wheel_angle = np.cumsum(-w * 3.0, axis=0) * 0.01
np.savez(a.out, t=t, x=bx, y=by, yaw=yaw, q_hfe=q, w_track=w, wheel_angle=wheel_angle, vx=vx, wz=wz)
print('saved', a.out, len(t), 'samples', round(t[0], 1), '-', round(t[-1], 1), 's')
