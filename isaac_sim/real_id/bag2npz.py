"""Convert a traquad run (robot bag + Qualisys bag) to one npz with plain arrays.

usage: python bag2npz.py <run_dir> <out.npz> [<pi3hat_moteus_int_msgs/msg dir>]
(run_dir = extracted archive of the traquad_bags release, with robot/ and qualisys/; needs pip rosbags)
"""
import os
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# message definitions of the robot (mulsbc_ws/src/pi3hat/pi3hat_moteus_int_msgs/msg), from MSG_DIR or the argument
MSG_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(os.environ.get('MSG_DIR', 'pi3hat_moteus_int_msgs/msg'))

store = get_typestore(Stores.ROS2_HUMBLE)
add = {}
for f in MSG_DIR.glob('*.msg'):
    add.update(get_types_from_msg(f.read_text(), f'pi3hat_moteus_int_msgs/msg/{f.stem}'))
store.register(add)

run = Path(sys.argv[1])
robot_dirs = sorted(p for p in (run / 'robot').iterdir() if p.is_dir())
qtm = sorted((run / 'qualisys').glob('*.db3'))
out = {}


def stamp(msg, t_rec):
    h = getattr(msg, 'header', None)
    if h is not None and (h.stamp.sec or h.stamp.nanosec):
        return h.stamp.sec + h.stamp.nanosec * 1e-9
    return t_rec * 1e-9


# main robot bag = the longest one (the joystick copy only has the mission topics)
def size(p):
    return sum(f.stat().st_size for f in p.glob('*.db3'))


robot = max(robot_dirs, key=size)
print('robot bag', robot.name)
with AnyReader([robot], default_typestore=store) as r:
    print({c.topic: c.msgcount for c in r.connections})
    data = {}
    for con, t, raw in r.messages():
        msg = r.deserialize(raw, con.msgtype)
        data.setdefault(con.topic, []).append((t, msg))

t0 = None


def joints(topic, fields, key):
    rows = data.get(topic, [])
    if not rows:
        return
    names = list(rows[0][1].name)
    out[key + '_names'] = np.array(names)
    out[key + '_t'] = np.array([tr * 1e-9 for tr, m in rows])
    out[key + '_tstamp'] = np.array([stamp(m, tr) for tr, m in rows])
    for f in fields:
        arr = np.full((len(rows), len(names)), np.nan)
        for i, (tr, m) in enumerate(rows):
            if list(m.name) != names:
                idx = [list(m.name).index(n) if n in m.name else -1 for n in names]
                v = getattr(m, f)
                arr[i] = [v[j] if j >= 0 and j < len(v) else np.nan for j in idx]
            else:
                v = np.asarray(getattr(m, f), float)
                arr[i, :len(v)] = v
        out[f'{key}_{f}'] = arr


joints('/omni_controller/joints_state', ['position', 'velocity', 'effort', 'temperature'] , 'js')
joints('/omni_controller/debug/joints_command', ['position', 'velocity', 'effort', 'kp_scale', 'kd_scale'], 'cmd')
joints('/omni_controller/joints_reference', ['position', 'velocity', 'effort', 'kp_scale', 'kd_scale'], 'ref')
joints('/omni_controller/direct_wheels_cmd', ['velocity', 'kp_scale', 'kd_scale'], 'dw')

rows = data.get('/omni_controller/twist_cmd', [])
if rows:
    out['tw_t'] = np.array([tr * 1e-9 for tr, m in rows])
    out['tw'] = np.array([[m.linear.x, m.linear.y, m.angular.z] for tr, m in rows])
rows = data.get('/imu_sensor/imu', [])
if rows:
    out['imu_t'] = np.array([tr * 1e-9 for tr, m in rows])
    out['imu_w'] = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for tr, m in rows])
    out['imu_q'] = np.array([[m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w] for tr, m in rows])
    out['imu_a'] = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for tr, m in rows])
rows = data.get('/joy', [])
if rows:
    n_ax = max(len(m.axes) for _, m in rows)
    n_bt = max(len(m.buttons) for _, m in rows)
    out['joy_t'] = np.array([tr * 1e-9 for tr, m in rows])
    out['joy_axes'] = np.array([list(m.axes) + [0] * (n_ax - len(m.axes)) for _, m in rows], float)
    out['joy_buttons'] = np.array([list(m.buttons) + [0] * (n_bt - len(m.buttons)) for _, m in rows], float)
rows = data.get('/omni_mulinex_joystick/identification/segment', [])
if rows:
    out['seg_t'] = np.array([tr * 1e-9 for tr, m in rows])
    out['seg_json'] = np.array([m.data for tr, m in rows])
for top in data:
    if top not in ('/omni_controller/joints_state', '/omni_controller/debug/joints_command',
                   '/omni_controller/joints_reference', '/omni_controller/direct_wheels_cmd',
                   '/omni_controller/twist_cmd', '/imu_sensor/imu', '/joy',
                   '/omni_mulinex_joystick/identification/segment'):
        print('skipped topic', top, len(data[top]), type(data[top][0][1]).__name__)

if qtm:
    import sqlite3
    db = sqlite3.connect(str(qtm[0]))
    topics = {i: (n, ty) for i, n, ty in db.execute('select id, name, type from topics')}
    print('qualisys topics', topics)
    tid = [i for i, (n, ty) in topics.items() if n.endswith('/pose')][0]
    rows = []
    for t, raw in db.execute('select timestamp, data from messages where topic_id=? order by timestamp', (tid,)):
        m = store.deserialize_cdr(raw, topics[tid][1])
        p, q = m.pose.position, m.pose.orientation
        rows.append((t * 1e-9, stamp(m, t), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    a = np.array(rows)
    out['qtm_t'] = a[:, 0]
    out['qtm_tstamp'] = a[:, 1]
    out['qtm_p'] = a[:, 2:5]
    out['qtm_q'] = a[:, 5:9]

np.savez_compressed(sys.argv[2], **out)
print('saved', sys.argv[2], {k: v.shape for k, v in out.items()})
