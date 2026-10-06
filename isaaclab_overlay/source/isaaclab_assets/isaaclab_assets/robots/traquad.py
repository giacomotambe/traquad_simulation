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
* ``body_{left,right}_{F,H}_ankle``: passive (USD damping)
* ``joint_wheel_{1..4}_{LEFT,RIGHT}_{F,H}``: track wheels, velocity controlled
* ``wheel_{1..4}_{LEFT,RIGHT}_{F,H}_roller_{0..7}_joint``: passive rollers (USD damping and dry friction)
"""

import os

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
        # the base settles at z = 0.245 m in this stance
        pos=(0.0, 0.0, 0.27),
        joint_pos={
            # |HFE| + |ankle| = pi/2 keeps the tracks flat on the ground
            "LF_HFE": 1.13,
            "LH_HFE": -1.13,
            "RF_HFE": -1.13,
            "RH_HFE": 1.13,
            # ankle limits: front in [0.091, 0.791], hind in [-0.791, -0.091]
            "body_.*_F_ankle": 0.44,
            "body_.*_H_ankle": -0.44,
            "joint_wheel_.*": 0.0,
            ".*_roller_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HFE"],
            joint_effort_limit=5.0,
            joint_velocity_limit=5.0,
            # same PD as Gazebo and the Isaac Sim tests
            stiffness=100.0,
            damping=10.0,
        ),
        # velocity control: zero stiffness, the damping is the velocity gain. Track motor (same as
        # isaac_sim/finalize_usd.py): 40 Nm and 2 Nm s/rad per track at r = 15 mm, split over its 4 wheels
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            joint_effort_limit=10.0,
            joint_velocity_limit=200.0,
            stiffness=0.0,
            damping=0.5,
            # the wheels are tiny (r = 1.5 cm): armature improves the solver conditioning
            armature=0.001,
        ),
        # ankles and rollers have no actuator: they keep the passive drives written in the USD
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
        # only wheel 2 of each track is driven: the whole track motor on it
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_2_.*"],
            joint_effort_limit=40.0,
            joint_velocity_limit=200.0,
            stiffness=0.0,
            damping=2.0,
            armature=0.001,
        ),
    },
)
"""Configuration of the TraQuad robot with cylinder wheels coupled by mimic joints."""
