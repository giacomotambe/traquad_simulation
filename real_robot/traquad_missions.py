#!/usr/bin/env python3
"""Identification missions for the real TraQuad (run on the NUC, ROS 2 Humble, mulinex_app workspace sourced).

The robot runs the low-level interface of mulsbc_ws:
  ros2 launch pi3hat_hw_interface moteus_pi3hat_interface.launch.py urdf_file:=traquad8_dji.xacro \
      conf_file:=traquad8_dji.yaml
and nothing else may publish on /omni_controller/twist_cmd while a mission runs (stop the joystick node after
activation and stand, or use --bringup here). Every mission is a fixed command sequence, so the same sequence can be
replayed in the simulation (isaac_sim/track_test.py runs the 'maneuver' one).

Commands go to the omni_controller topics:
- twist missions: /omni_controller/twist_cmd (BestEffort, deadline 105 ms as the controller requests; IK mode,
  TraquadIK of traquad8_dji.yaml, wheel_rad 0.045, 2 Hz low-pass on the twist);
- wheel missions: /omni_controller/direct_wheels_cmd after set_wheel_mode true (track joint speeds in rad/s, no
  low-pass; right joints turn the other way for forward motion, as in the IK);
- hip missions: /omni_controller/joints_reference (HFE positions, kp_scale = kd_scale = 1: Kp 60, Kd 0.6).
Commands are published at 100 Hz; the controller zeroes the tracks and holds the hips after 0.5 s without them.

usage: traquad_missions.py <mission> [--bag DIR] [--bringup] [--list]
"""
import argparse
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from pi3hat_moteus_int_msgs.msg import JointsCommand
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool

RATE = 100.0
WHEELS = ['LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL']
WHEEL_SIGN = {'LF_WHEEL': 1.0, 'LH_WHEEL': 1.0, 'RF_WHEEL': -1.0, 'RH_WHEEL': -1.0}   # forward motion
HFES = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE']
HFE_SIGN = {'LF_HFE': 1.0, 'LH_HFE': -1.0, 'RF_HFE': -1.0, 'RH_HFE': 1.0}             # q_stand = sign * 1.2
STAND = 1.2
TOPICS = ['/omni_controller/joints_state', '/omni_controller/debug/joints_command', '/omni_controller/twist_cmd',
          '/omni_controller/direct_wheels_cmd', '/omni_controller/joints_reference', '/imu_sensor/imu',
          '/omni_controller/distributors_state', '/omni_controller/performance']


def twist(*segments):
    """Segments (duration [s], vx [m/s], wz [rad/s])."""
    return 'twist', list(segments)


def wheels(*segments):
    """Segments (duration [s], left joint speed [rad/s], right joint speed [rad/s], kd_scale), forward positive."""
    return 'wheels', list(segments)


def hips(*segments):
    """Segments (duration [s], |HFE| [rad]), same angle on the four legs."""
    return 'hips', list(segments)


def ramp(t_ramp, t_hold, vx, wz=0.0, steps=20):
    return [(t_ramp / steps, vx * (i + 1) / steps, wz * (i + 1) / steps) for i in range(steps)] + [(t_hold, vx, wz)]


MISSIONS = {
    # M4/M5, same as track_test.py: 3 s settle, 8 s in place at 0.5 rad/s, 2 s stop, 10 s turn 0.2 m/s + 0.5 rad/s,
    # 2 s stop, 5 s straight 0.2 m/s (the outer track asks 0.30 m/s, at the 0.28 m/s speed limit)
    'maneuver': twist((3, 0, 0), (8, 0, 0.5), (2, 0, 0), (10, 0.2, 0.5), (2, 0, 0), (5, 0.2, 0), (3, 0, 0)),
    # M3: straight runs, belt radius and rolling drag (mark start and end on the floor)
    'straight': twist(*[s for v in (0.05, 0.1, 0.2, 0.25) for s in ((2, 0, 0), (6, v, 0), (2, 0, 0), (6, -v, 0))],
                      (2, 0, 0)),
    # M4: rotation in place at three rates, both directions
    'spin': twist(*[s for w in (0.25, 0.5, 1.0) for s in ((2, 0, 0), (8, 0, w), (2, 0, 0), (8, 0, -w))], (2, 0, 0)),
    # M5: turns of different radius at 0.15 m/s
    'turns': twist(*[s for w in (0.25, -0.25, 0.5, -0.5) for s in ((2, 0, 0), (10, 0.15, w))], (2, 0, 0)),
    # M6: hold still with the tracks at zero speed (velocity loop only: pushes make it creep), 30 s
    'hold': twist((30, 0, 0)),
    # M7: drive slowly into a wall and push for 5 s
    'wall': twist((2, 0, 0), *ramp(2, 5, 0.1), (3, 0, 0)),
    # M8: climb a board: 8 s at 0.1 m/s, 5 s stopped on the slope, 8 s back down
    'climb': twist((2, 0, 0), *ramp(1, 7, 0.1), (5, 0, 0), *ramp(1, 7, -0.1), (2, 0, 0)),
    # M1a, robot on a stand with the tracks in the air: track joint speed steps (10 rad/s is the speed limit)
    'wheel_steps': wheels(*[s for w in (1, 3, 6, 10, 12) for s in ((2, 0, 0, 1), (3, w, w, 1), (2, 0, 0, 1),
                                                                   (3, -w, -w, 1))], (2, 0, 0, 1)),
    # M1b, tracks in the air: very slow speed, the frame turns with the sprocket until its stops (film it)
    'ankle_sweep': wheels((2, 0, 0, 1), (8, 0.3, 0.3, 1), (2, 0, 0, 1), (8, -0.3, -0.3, 1), (2, 0, 0, 1)),
    # M1c, tracks in the air: motors free (kd_scale 0) for 20 s; tilt a track by hand and release it (film it)
    'ankle_free': wheels((20, 0, 0, 0)),
    # M1d, robot on a stand with the legs free: hip position steps around the stand pose
    'hip_steps': hips((2, STAND), (2, STAND + 0.1), (2, STAND), (2, STAND - 0.1), (2, STAND), (2, STAND + 0.3),
                      (2, STAND)),
}


class MissionNode(Node):
    def __init__(self):
        super().__init__('traquad_missions')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, deadline=Duration(seconds=0.1))
        self.twist_pub = self.create_publisher(Twist, '/omni_controller/twist_cmd', qos)
        self.wheel_pub = self.create_publisher(JointsCommand, '/omni_controller/direct_wheels_cmd', 5)
        self.hip_pub = self.create_publisher(JointsCommand, '/omni_controller/joints_reference', 5)

    def call(self, name, value=True):
        cli = self.create_client(SetBool, '/omni_controller/' + name)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'service /omni_controller/{name} not available')
        fut = cli.call_async(SetBool.Request(data=value))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        self.get_logger().info(f'{name}({value}): {fut.result()}')

    def send(self, kind, seg):
        if kind == 'twist':
            msg = Twist()
            msg.linear.x, msg.angular.z = float(seg[1]), float(seg[2])
            self.twist_pub.publish(msg)
        elif kind == 'wheels':
            msg = JointsCommand(name=WHEELS)
            msg.velocity = [WHEEL_SIGN[n] * float(seg[1] if n.startswith('L') else seg[2]) for n in WHEELS]
            msg.kp_scale = [0.0] * 4
            msg.kd_scale = [float(seg[3])] * 4
            self.wheel_pub.publish(msg)
        else:
            msg = JointsCommand(name=HFES)
            msg.position = [HFE_SIGN[n] * float(seg[1]) for n in HFES]
            msg.velocity = [0.0] * 4
            msg.effort = [0.0] * 4
            msg.kp_scale = [1.0] * 4
            msg.kd_scale = [1.0] * 4
            self.hip_pub.publish(msg)

    def run(self, kind, segments):
        t0 = time.monotonic()
        for seg in segments:
            self.get_logger().info(f't {time.monotonic() - t0:6.2f} s  {kind} {seg[1:]} for {seg[0]:g} s')
            end = time.monotonic() + seg[0]
            while time.monotonic() < end:
                self.send(kind, seg)
                time.sleep(1.0 / RATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mission', nargs='?', choices=sorted(MISSIONS))
    parser.add_argument('--bag', help='record the mission topics into this rosbag directory (mcap)')
    parser.add_argument('--bringup', action='store_true', help='call activate_srv and stand_srv first (6 s)')
    parser.add_argument('--list', action='store_true', help='print the missions and their duration')
    args = parser.parse_args()
    if args.list or args.mission is None:
        for name, (kind, segs) in MISSIONS.items():
            print(f'{name:12s} {kind:6s} {sum(s[0] for s in segs):6.1f} s')
        return
    kind, segments = MISSIONS[args.mission]
    rclpy.init()
    node = MissionNode()
    bag = None
    try:
        if args.bag:
            bag = subprocess.Popen(['ros2', 'bag', 'record', '-s', 'mcap', '-o', args.bag] + TOPICS)
            time.sleep(2.0)
        if args.bringup:
            node.call('activate_srv')
            node.call('stand_srv')
            time.sleep(6.0)
        if kind == 'wheels':
            node.call('set_wheel_mode', True)
        node.run(kind, segments)
    finally:
        if kind == 'wheels':
            node.call('set_wheel_mode', False)
        if bag:
            time.sleep(1.0)
            bag.send_signal(signal.SIGINT)
            bag.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
