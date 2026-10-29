# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TraQuad robot (4 legs with a track each, the track simulated by 4 roller wheels).

The following configuration parameters are available:

* :obj:`TRAQUAD_CFG`: TraQuad with roller wheels and real masses (8.0 kg)

The USD is the instanceable asset of the traquad_simulation repository (``isaac_sim/assets/traquad``, rebuilt from
the xacro by ``isaac_sim/make_isaac_asset.sh``); the ``TRAQUAD_USD`` environment variable overrides its path.
It needs the PhysX backend: the roller dry friction is a PhysX joint attribute.

Joints:

* ``{LF,LH,RF,RH}_HFE``: hip flexion, position controlled
* ``body_{left,right}_{F,H}_ankle``: track pivot, passive on the ground, velocity controlled in the air (see below)
* ``joint_wheel_{1..4}_{LEFT,RIGHT}_{F,H}``: track wheels, velocity controlled
* ``wheel_{1..4}_{LEFT,RIGHT}_{F,H}_roller_{0..7}_joint``: passive rollers (USD damping and dry friction)
"""

import os

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# isaaclab_traquad/source/isaaclab_assets/isaaclab_assets/robots/traquad.py -> traquad_simulation
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))
TRAQUAD_USD_PATH = os.environ.get(
    "TRAQUAD_USD", os.path.join(_REPO_DIR, "isaac_sim", "assets", "traquad", "traquad.usda")
)
"""Path of the TraQuad USD (roller wheels)."""

##
# Ankle model
##

# On the real robot the track motor sits on the leg and drives a sprocket coaxial with the ankle; the track frame
# pivots freely on the same axis, between mechanical end stops. On the ground the belt is held by the ground and moves
# the robot; the ankle is free, but the sprocket drives the belt from the pivot: in the simulation the motor drives the
# track wheels and its reaction stays in the track frame, so the track action applies to the ankle the torque the
# sprocket puts on the track, (SPROCKET_RADIUS / WHEEL_RADIUS) x the drive torque of the track wheels. Without it the
# traction pitches the track with F * h instead of F * (h - r_s), and the tracks tip on slopes the real robot climbs
# (35 deg on wood). In the air belt and frame turn with the sprocket: the track action then drives the ankle with the
# controller of the track and the same speed command, w_target = v / SPROCKET_RADIUS: the track Kd as damping of the
# implicit ankle drive (ANKLE_SPEED_GAIN, velocity target w_target) and the integral of the wheel compensation of the
# robot (ANKLE_INTEGRAL_GAIN, limited to +-ANKLE_MAX_TORQUE) as effort. On the ground (contact sensor) the drive goes
# back to the passive damping ANKLE_PASSIVE_DAMPING with target 0, the effort is the sprocket torque and the integral is
# reset.
WHEEL_RADIUS = 0.015
"""Radius of the simulated track wheels [m]."""
SPROCKET_RADIUS = 0.045
"""Radius of the drive sprocket on the ankle axis [m]: belt speed per rad/s of the track joint, ``wheel_rad`` of the
differential IK of the real robot (mulsbc_ws traquad8_dji.yaml, measured with the Qualisys on 2026-10-22; was 0.028)."""
SPROCKET_TORQUE_FACTOR = 0.0
"""Scale of the sprocket torque on the ankle (1 = the model above). Off since 2026-10-26: with r_s = 45 mm it tips the
tracks onto their stops on flat ground (isaac_sim/track_test.py: 4% of the commanded yaw rate, no straight driving)
and the replays of the real runs match the Qualisys without it (isaac_sim/real_id). Without it the robot climbs 30 deg
at mu 0.6 but not 35 deg at mu 0.75 (isaac_sim/README.md)."""
TRACK_FRICTION = 1.16
"""Coulomb friction inside a track (belt, sprocket, gear) at the track joint [N m], from the torque of the real track
drives on the straight runs of 2026-10-23 (isaac_sim/real_id)."""
TRACK_VISCOUS = 0.013
"""Viscous friction inside a track at the track joint [N m s/rad] (same runs)."""
HIP_MAX_TORQUE = 10.0
"""Torque limit of a hip motor [N m] (not from the robot: raised from 5 N m, which saturated in training). Hips and tracks
use the same actuator (same K_t), with moteus max_current_A 80 / 40 A and gear ratio 9 / 15 (mulsbc_ws
traquad8_dji.xacro): set it from K_t x 80 A x 9 when K_t is known."""
TRACK_MAX_TORQUE = HIP_MAX_TORQUE * (40.0 * 15.0) / (80.0 * 9.0)
"""Torque limit of a track motor at the sprocket [N m]: same actuator as the hips, 40 A x gear 15 against 80 A x gear 9,
0.83 x HIP_MAX_TORQUE."""
ANKLE_PASSIVE_DAMPING = 0.01
"""Passive viscous damping of the ankle pivot [N m s/rad] (same value as the USD ankle drive)."""
ANKLE_SPEED_GAIN = 0.9
"""Damping of the ankle drive while the track is in the air [N m s/rad]: Kd of the track joint (moteus, mulsbc_ws
traquad8_dji.xacro). Until 2026-10-27 the ankle had a weak PI (0.2 / 2 / 0.05 N m) in the air."""
# Wheel compensation of the real track controller (mulsbc_ws traquad8_dji.yaml, omni_controller wheel_compensation,
# enabled on the robot; run lowlevel_wheel_compensation of 2026-10-23): the track drives have Kd only, so the omni_controller
# adds as effort a friction feedforward and the integral of the track speed error. The track action applies it to the
# wheels on top of their implicit drive (the Kd), at the track joint and referred to the wheels.
WHEEL_COMP_FRICTION = 1.4
"""Friction feedforward of the wheel compensation [N m] at the track joint."""
WHEEL_COMP_VISCOUS = 0.017
"""Viscous feedforward of the wheel compensation [N m s/rad] at the track joint."""
WHEEL_COMP_FADE = 1.0
"""Track speed command below which the feedforward fades to zero [rad/s]."""
WHEEL_COMP_KI = 3.0
"""Integral gain of the wheel compensation [N m/rad]."""
WHEEL_COMP_I_MAX = 1.0
"""Limit of the integral term of the wheel compensation [N m]."""
ANKLE_INTEGRAL_GAIN = WHEEL_COMP_KI
"""Integral gain of the ankle in the air [N m/rad]: ki of the wheel compensation of the omni_controller."""
ANKLE_MAX_TORQUE = WHEEL_COMP_I_MAX
"""Limit of the integral term of the ankle in the air [N m]: i_max of the wheel compensation."""
ANKLE_VELOCITY_RATIO = {"LEFT": -WHEEL_RADIUS / SPROCKET_RADIUS, "RIGHT": WHEEL_RADIUS / SPROCKET_RADIUS}
"""Ankle velocity target per unit of wheel velocity target, by side. The wheel axes point along -y (left) and +y
(right), the ankle axes of both sides along +y: the sprocket turns with the wheels."""
ANKLE_SPROCKET_TORQUE_RATIO = {
    "LEFT": -SPROCKET_TORQUE_FACTOR * SPROCKET_RADIUS / WHEEL_RADIUS,
    "RIGHT": SPROCKET_TORQUE_FACTOR * SPROCKET_RADIUS / WHEEL_RADIUS,
}
"""Ankle torque per unit of summed wheel drive torque of the track, by side (sprocket torque on the track)."""

##
# Configuration
##

TRAQUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAQUAD_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # self-collisions off: the rollers overlap the wheel bodies
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
        ),
        fix_root_link=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # the base settles at z = 0.262 m in this stance
        pos=(0.0, 0.0, 0.29),
        joint_pos={
            # |HFE| + |ankle| = pi/2 keeps the tracks flat on the ground. HFE 1.47 (legs almost vertical): the hips
            # need about 1.5 N m to hold the robot, against 3.2-3.6 N m at 1.13
            "LF_HFE": 1.47,
            "LH_HFE": -1.47,
            "RF_HFE": -1.47,
            "RH_HFE": 1.47,
            # ankle end stops (relative to the upper leg): +-30 deg around the flat pose, front in [-0.423, 0.624],
            # hind in [-0.624, 0.423]
            "body_.*_F_ankle": 0.10,
            "body_.*_H_ankle": -0.10,
            "joint_wheel_.*": 0.0,
            ".*_roller_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HFE"],
            joint_effort_limit=HIP_MAX_TORQUE,
            # real robot (mulsbc_ws traquad8_dji.xacro, joint units): moteus max_velocity 10 rad/s, Kp 60, Kd 0.6
            joint_velocity_limit=10.0,
            stiffness=60.0,
            damping=0.6,
        ),
        # velocity control: zero stiffness, the damping is the velocity gain. Track motor (same as
        # isaac_sim/finalize_usd.py): TRACK_MAX_TORQUE at the sprocket, referred to the 15 mm wheels and split over 4.
        # Velocity gain of the real track joint, Kd 0.9 N m s/rad at the sprocket (moteus, Kp 0), referred to the
        # wheels: 0.9 * (WHEEL_RADIUS / SPROCKET_RADIUS)^2 / 4 per wheel (the same belt force per m/s of speed error)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            joint_effort_limit=TRACK_MAX_TORQUE * WHEEL_RADIUS / SPROCKET_RADIUS / 4,
            joint_velocity_limit=200.0,
            stiffness=0.0,
            damping=0.9 * (WHEEL_RADIUS / SPROCKET_RADIUS) ** 2 / 4,
            # internal friction of the track, referred to the wheels and split over 4 (same values as the USD)
            friction=TRACK_FRICTION * WHEEL_RADIUS / SPROCKET_RADIUS / 4,
            dynamic_friction=TRACK_FRICTION * WHEEL_RADIUS / SPROCKET_RADIUS / 4,
            viscous_friction=TRACK_VISCOUS * (WHEEL_RADIUS / SPROCKET_RADIUS) ** 2 / 4,
            # the wheels are tiny (r = 1.5 cm): armature improves the solver conditioning
            armature=0.001,
        ),
        # ankle: passive viscous damping (solver) + the effort of the PI velocity controller computed by the track
        # action on every physics step (zero on the ground), clipped here to ANKLE_MAX_TORQUE
        # ankle: implicit velocity drive, passive damping on the ground; the track action switches it to the track Kd
        # with the sprocket speed as target while the track is in the air, and adds the integral (or the sprocket
        # torque on the ground) as effort
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["body_.*_ankle"],
            stiffness=0.0,
            damping=ANKLE_PASSIVE_DAMPING,
            joint_effort_limit=1.2 * TRACK_MAX_TORQUE,
            joint_velocity_limit=100.0,
        ),
        # rollers have no actuator: they keep the passive drives written in the USD
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of the TraQuad robot with roller wheels."""
