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
        self.future_num_frames = 1 # RYAN : to get gamma, just need 1 step forward with no dynamics
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


        self.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        self.cbf.type = "backup_barrier_cbf"
        self.cbf.T_horizon = 3
        self.cbf.alpha = 0.2
        self.cbf.veh_veh = True
        self.cbf.normalize_constraint = True

        self.scene_centric = True

        self.loss_weights.constraint_loss = 1.0
        self.loss_weights.max_likelihood_loss = 0.0
        self.loss_weights.sum_resp_loss = 10

        self.constraint_loss.leaky_relu_negative_slope = 1e-1
        self.sum_resp_loss.leaky_relu_negative_slope = 0

        self.max_angle_diff = 60


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


class SpatialPlannerConfig(BehaviorCloningConfig):
    def __init__(self):
        super(SpatialPlannerConfig, self).__init__()
        self.eval_class = None

        self.name = "spatial_planner"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.loss_weights.pixel_res_loss = 1.0
        self.loss_weights.pixel_yaw_loss = 1.0


class AgentPredictorConfig(BehaviorCloningConfig):
    def __init__(self):
        super(AgentPredictorConfig, self).__init__()
        self.eval_class = "HierAgentAware"

        self.name = "agent_predictor"
        self.agent_feature_dim = 128
        self.global_feature_dim = 128
        self.context_size = (30, 30)
        self.goal_conditional = True
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128, 128)

        self.use_rotated_roi = False
        self.use_transformer = False
        self.roi_layer_key = "layer2"
        self.use_GAN = False
        self.history_conditioning = False

        self.loss_weights.lane_reg_loss = 0.5
        self.loss_weights.GAN_loss = 0.5

        self.optim_params.GAN.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.GAN.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.GAN.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.GAN.regularization.L2 = 0.00  # L2 regularization strength


class BehaviorCloningGCConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningGCConfig, self).__init__()
        self.eval_class = None
        self.name = "bc_gc"
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128)


class EBMMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(EBMMetricConfig, self).__init__()
        self.eval_class = None
        self.name = "ebm"
        self.negative_source = "permute"
        self.map_feature_dim = 64
        self.traj_feature_dim = 32
        self.embedding_dim = 32
        self.embed_layer_dims = (128, 64)
        self.loss_weights.infoNCE_loss = 1.0


class OccupancyMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(OccupancyMetricConfig, self).__init__()
        self.eval_class = "metric"
        self.name = "occupancy"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.agent_future_cond.enabled = True
        self.agent_future_cond.every_n_frame = 5


class VAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(VAEConfig, self).__init__()
        self.eval_class = "TrafficSim"
        self.name = "vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        self.vae.latent_dim = 4
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)

        self.loss_weights.kl_loss = 1e-4


class DiscreteVAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(DiscreteVAEConfig, self).__init__()
        self.eval_class = "TPP"

        self.name = "discrete_vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        self.ego_conditioning = False
        self.EC_feat_dim = 64
        self.vae.latent_dim = 10
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = True
        self.vae.recon_loss_type = "NLL"
        self.vae.logpi_clamp = -6.0

        self.loss_weights.kl_loss = 10
        self.loss_weights.EC_coll_loss = 10
        self.loss_weights.deviation_loss = 0.5
        self.eval.mode = "mean"

        self.agent_future_cond.enabled = False
        self.agent_future_cond.feature_dim = 32
        self.agent_future_cond.transformer = True

        self.min_std = 0.1


class TreeVAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(TreeVAEConfig, self).__init__()
        self.eval_class = None

        self.name = "tree_vae"
        self.map_feature_dim = 256
        self.goal_conditional = True
        self.goal_feature_dim = 32
        self.stage = 2
        self.num_frames_per_stage = 10

        self.vae.latent_dim = 4
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = False
        self.vae.recon_loss_type = "MSE"
        self.vae.logpi_clamp = -6.0
        self.ego_conditioning = True
        self.EC_feat_dim = 64
        self.loss_weights.EC_collision_loss = 10
        self.loss_weights.deviation_loss = 0.5
        self.loss_weights.kl_loss = 10
        self.loss_weights.collision_loss = 5.0
        self.eval.mode = "sum"
        self.scene_centric = True
        self.shuffle = True
        self.vectorize_lane = False
        self.min_std = 0.1
        self.perturb.enabled=True
        self.perturb.N_pert = 3
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = 2.0
        self.perturb.OU.scale = [1.0,1.0,0.2]


class BehaviorCloningECConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningECConfig, self).__init__()
        self.eval_class = None

        self.name = "bc_ec"
        self.map_feature_dim = 256
        self.goal_conditional = True
        self.goal_feature_dim = 32

        self.EC.feature_dim = 64
        self.EC.RNN_hidden_size = 32
        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.yaw_reg_loss = 0.05
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 4
        self.loss_weights.EC_collision_loss = 5
        self.loss_weights.deviation_loss = 0.2


class GANConfig(BehaviorCloningConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self.eval_class = "GAN"

        self.name = "gan"

        self.map_feature_dim = 256
        self.optim_params.disc.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.policy.learning_rate.initial = 1e-4  # generator learning rate

        self.decoder.layer_dims = (128, 128)

        self.traj_encoder.rnn_hidden_size = 100
        self.traj_encoder.feature_dim = 32
        self.traj_encoder.mlp_layer_dims = (128, 128)

        self.gan.latent_dim = 4
        self.gan.loss_type = "lsgan"
        self.gan.disc_layer_dims = (128, 128)
        self.gan.num_eval_samples = 10

        self.loss_weights.prediction_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.0
        self.loss_weights.gan_gen_loss = 1.0
        self.loss_weights.gan_disc_loss = 1.0

        self.optim_params.disc.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.disc.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.disc.regularization.L2 = 0.00  # L2 regularization strength


class TransformerPredConfig(AlgoConfig):
    def __init__(self):
        super(TransformerPredConfig, self).__init__()

        self.name = "TransformerPred"
        self.model_architecture = "Factorized"
        self.history_num_frames = 10
        # this will also create raster history (we need to remove the raster from train/eval dataset - only visualization)
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 20
        self.step_time = 0.1
        self.N_t = 2
        self.N_a = 1
        self.d_model = 128
        self.XY_pe_dim = 16
        self.temporal_pe_dim = 16
        self.map_emb_dim = 32
        self.d_ff = 256
        self.head = 4
        self.dropout = 0.01
        self.XY_step_size = [2.0, 2.0]

        self.disable_other_agents = False
        self.disable_map = False
        self.goal_conditioned = True

        # self.disable_lane_boundaries = True
        self.global_head_dropout = 0.0
        self.training_num_N = 10000
        self.N_layer_enc = 2
        self.N_layer_tgt_enc = 1
        self.N_layer_tgt_dec = 1
        self.vmax = 30
        self.vmin = -10
        self.calc_likelihood = False
        self.calc_collision = False

        self.map_enc_mode = "all"
        self.temporal_bias = 0.5
        self.weights.lane_regulation_weight = 1.5
        self.weights.weights_scaling = [1.0, 1.0, 1.0]
        self.weights.ego_weight = 1.0
        self.weights.all_other_weight = 0.5
        self.weights.collision_weight = 1
        self.weights.reg_weight = 10
        self.weights.goal_reaching_weight = 2

        # map encoding parameters
        self.CNN.map_channels = (self.history_num_frames+1)*2+3
        self.CNN.lane_channel = (self.history_num_frames+1)*2
        self.CNN.hidden_channels = [20, 20, 20, 10]
        self.CNN.ROI_outdim = 10
        self.CNN.output_size = self.map_emb_dim
        self.CNN.patch_size = [15, 35, 25, 25]
        self.CNN.kernel_size = [5, 5, 5, 3]
        self.CNN.strides = [1, 1, 1, 1]
        self.CNN.input_size = [224, 224]
        self.CNN.veh_ROI_outdim = 4
        self.CNN.veh_patch_scale = 1.5

        # Multi-modal prediction
        self.M = 1

        self.Discriminator.N_layer_enc = 1

        self.vectorize_map = False
        self.vectorize_lane = True
        self.points_per_lane = 5

        self.try_to_use_cuda = True

        # self.model_params.future_num_frames = 0
        # self.model_params.step_time = 0.2
        self.render_ego_history = False
        # self.model_params.history_num_frames_ego = 0
        # self.model_params.history_num_frames = 0
        # self.model_params.history_num_frames_agents = 0

        self.optim_params.policy.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class TransformerGANConfig(TransformerPredConfig):
    def __init__(self):
        super(TransformerGANConfig, self).__init__()
        self.name = "TransformerGAN"
        self.calc_likelihood = True
        self.f_steps = 5
        self.weights.GAN_weight = 0.2
        self.GAN_static = True

        self.optim_params_discriminator.learning_rate.initial = (
            1e-3  # policy learning rate
        )
        self.optim_params_discriminator.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params_discriminator.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params_discriminator.regularization.L2 = (
            0.00  # L2 regularization strength
        )
