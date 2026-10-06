# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class TraQuadRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "traquad_rough"
    # asymmetric actor-critic: the critic also sees base linear velocity and height scan
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        # one unit of wheel action is 1.3 m/s of track speed: a lower initial noise avoids violent exploration
        distribution_cfg=RslRlRNNModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log"),
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TraQuadFlatPPORunnerCfg(TraQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "traquad_flat"
