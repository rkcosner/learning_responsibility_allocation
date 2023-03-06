import math

from tbsim.configs.base import AlgoConfig

class ResponsibilityConfig(AlgoConfig):
    def __init__(self):
        super(ResponsibilityConfig, self).__init__()
        self.eval_class = "Responsibility"

        self.name = "resp"
        self.model_architecture = "resnet18"
        self.map_feature_dim = 256 
        self.history_num_frames = 10
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 5 # RYAN : to get gamma, just need 1 step forward with no dynamics
        self.responsibility_dim = 1 # RYAN : gamma is the 1 dimensional projection onto dhdx 
        self.responsibility_dynamics = None # RYAN : we can force responsibility to have dynamics here
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = (128,128)
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        # self.loss_weights.prediction_loss = 1.0
        # self.loss_weights.goal_loss = 0.0
        # self.loss_weights.collision_loss = 0.0
        # self.loss_weights.yaw_reg_loss = 0.1


        self.optim_params.policy.learning_rate.initial = 1e-5  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        self.cbf.type = "backup_barrier_cbf"
        self.cbf.T_horizon = 1
        self.cbf.alpha = 0.5
        self.cbf.veh_veh = True
        self.cbf.normalize_constraint = False
        self.cbf.saturate_cbf = True
        self.cbf.backup_controller_type = "idle" 

        self.scene_centric = True

        self.loss_weights.constraint_loss = 1.0
        self.loss_weights.max_likelihood_loss = 0.1
        self.loss_weights.sum_resp_loss = 10

        self.constraint_loss.leaky_relu_negative_slope = 0.1
        self.sum_resp_loss.leaky_relu_negative_slope = 0.01

        self.max_angle_diff = 100 # max angle diff away from 0 to consider, in degrees


class BehaviorCloningConfig(AlgoConfig):
    def __init__(self):
        super(BehaviorCloningConfig, self).__init__()
        self.eval_class = "BC"

        self.name = "bc"
        self.model_architecture = "resnet18"
        self.map_feature_dim = 256
        self.history_num_frames = 10
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 20
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.1

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength