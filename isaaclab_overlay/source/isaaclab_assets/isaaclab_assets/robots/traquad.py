# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TraQuad robot (4 legs with a track each, the track simulated by 4 roller wheels).

The following configuration parameters are available:

* :obj:`TRAQUAD_CFG`: TraQuad with roller wheels and real masses (7.8 kg)
* :obj:`TRAQUAD_CYLINDER_CFG`: TraQuad with plain cylinder wheels: wheel 2 of each track is driven, wheels 1, 3, 4
  follow it through mimic joints

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
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import replace

# isaaclab_traquad/source/isaaclab_assets/isaaclab_assets/robots/traquad.py -> traquad_simulation
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))
TRAQUAD_USD_PATH = os.environ.get(
    "TRAQUAD_USD", os.path.join(_REPO_DIR, "isaac_sim", "assets", "traquad", "traquad.usda")
)
"""Path of the TraQuad USD (roller wheels)."""
TRAQUAD_CYLINDER_USD_PATH = os.path.join(_REPO_DIR, "isaac_sim", "assets", "traquad_cylinder", "traquad.usda")
"""Path of the TraQuad USD with cylinder wheels (``TRACK_MODEL=cylinder isaac_sim/make_isaac_asset.sh``)."""

##
# Ankle model
##

# On the real robot the track motor sits on the leg and drives a sprocket coaxial with the ankle; the track frame
# pivots freely on the same axis, between mechanical end stops. On the ground the ankle is passive (the belt is held
# by the ground and moves the robot). In the air belt and frame turn with the sprocket: the track action then also
# velocity-controls the ankle at the speed that gives the commanded belt speed, w_target = v / SPROCKET_RADIUS, with a
# PI controller (the integral removes the steady error due to gravity: the track keeps turning at w_target, however
# small, until an end stop):
#   tau_ankle = clip(ANKLE_SPEED_GAIN * e + ANKLE_INTEGRAL_GAIN * int(e dt), +-ANKLE_MAX_TORQUE),  e = w_target - dq
#   tau_ankle = 0 and integral reset when a body of the track touches something (contact sensor of the task)
# plus a passive viscous damping ANKLE_PASSIVE_DAMPING, always (physics solver).
WHEEL_RADIUS = 0.015
"""Radius of the simulated track wheels [m]."""
SPROCKET_RADIUS = 0.015
"""Radius of the drive sprocket on the ankle axis [m] (placeholder until measured on the robot)."""
ANKLE_PASSIVE_DAMPING = 0.01
"""Passive viscous damping of the ankle pivot [N m s/rad] (same value as the USD ankle drive)."""
ANKLE_SPEED_GAIN = 0.2
"""Proportional gain of the ankle velocity controller used when the track is in the air [N m s/rad]."""
ANKLE_INTEGRAL_GAIN = 2.0
"""Integral gain of the ankle velocity controller [N m/rad]."""
ANKLE_MAX_TORQUE = 0.05
"""Largest torque of the ankle velocity controller [N m] (gravity on a hanging track is at most about 0.034 N m)."""
ANKLE_VELOCITY_RATIO = {"LEFT": -WHEEL_RADIUS / SPROCKET_RADIUS, "RIGHT": WHEEL_RADIUS / SPROCKET_RADIUS}
"""Ankle velocity target per unit of wheel velocity target, by side. The wheel axes point along -y (left) and +y
(right), the ankle axes of both sides along +y: the sprocket turns with the wheels."""

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
            # hip motor limit raised from 5 N m (Gazebo value): the hips saturated in the first trainings
            joint_effort_limit=10.0,
            joint_velocity_limit=5.0,
            # same PD as Gazebo and the Isaac Sim tests
            stiffness=100.0,
            damping=10.0,
        ),
        # velocity control: zero stiffness, the damping is the velocity gain. Track motor (same as
        # isaac_sim/finalize_usd.py): 1.5 Nm and 2 Nm s/rad per track at r = 15 mm (100 N of belt force), split over
        # its 4 wheels. More torque made the tracks tip onto their end stops in turns (traction below the ankle)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            joint_effort_limit=0.375,
            joint_velocity_limit=200.0,
            stiffness=0.0,
            damping=0.5,
            # the wheels are tiny (r = 1.5 cm): armature improves the solver conditioning
            armature=0.001,
        ),
        # ankle: passive viscous damping (solver) + the effort of the PI velocity controller computed by the track
        # action on every physics step (zero on the ground), clipped here to ANKLE_MAX_TORQUE
        "ankles": IdealPDActuatorCfg(
            joint_names_expr=["body_.*_ankle"],
            stiffness=0.0,
            damping=0.0,
            actuator_effort_limit=ANKLE_MAX_TORQUE,
            joint_velocity_limit=100.0,
            viscous_friction=ANKLE_PASSIVE_DAMPING,
        ),
        # rollers have no actuator: they keep the passive drives written in the USD
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of the TraQuad robot with roller wheels."""

TRAQUAD_CYLINDER_CFG = replace(
    TRAQUAD_CFG,
    spawn=replace(TRAQUAD_CFG.spawn, usd_path=TRAQUAD_CYLINDER_USD_PATH),
    # no roller joints
    init_state=replace(
        TRAQUAD_CFG.init_state,
        joint_pos={k: v for k, v in TRAQUAD_CFG.init_state.joint_pos.items() if "roller" not in k},
    ),
    actuators={
        "legs": TRAQUAD_CFG.actuators["legs"],
        "ankles": TRAQUAD_CFG.actuators["ankles"],
        # only wheel 2 of each track is driven: the whole track motor on it
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_2_.*"],
            joint_effort_limit=1.5,
            joint_velocity_limit=200.0,
            stiffness=0.0,
            damping=2.0,
            armature=0.001,
        ),
    },
)
"""Configuration of the TraQuad robot with cylinder wheels coupled by mimic joints."""
