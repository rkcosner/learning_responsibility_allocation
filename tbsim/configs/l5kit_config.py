import math

from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig

MAX_POINTS_LANE = 5


class L5KitTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitTrainConfig, self).__init__()

        self.dataset_path = "YOUR_DAFA_FOLDER"
        self.dataset_valid_key = "scenes/validate.zarr"
        self.dataset_train_key = "scenes/train.zarr"
        self.dataset_meta_key = "meta.json"
        self.datamodule_class = "L5RasterizedDataModule"

        self.rollout.enabled = True
        self.rollout.every_n_steps = 500
        self.rollout.num_episodes = 10
        self.rollout.num_scenes = 3
        self.rollout.n_step_action = 10

        ## training config
        self.training.batch_size = 32
        self.training.num_steps = 200000
        self.training.num_data_workers = 4

        self.save.every_n_steps = 1000

        ## validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 100


class L5KitMixedTrainConfig(L5KitTrainConfig):
    def __init__(self):
        super(L5KitMixedTrainConfig, self).__init__()
        self.datamodule_class = "L5MixedDataModule"


class L5KitEnvConfig(EnvConfig):
    def __init__(self):
        super(L5KitEnvConfig, self).__init__()

        self.name = "l5_rasterized"

        # raster image size [pixels]
        self.rasterizer.raster_size = (224, 224)

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = (0.5, 0.5)

        # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        self.rasterizer.ego_center = (0.25, 0.5)

        self.rasterizer.map_type = "py_semantic"

        # the keys are relative to the dataset environment variable
        self.rasterizer.satellite_map_key = "aerial_map/aerial_map.png"
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        self.rasterizer.filter_agents_threshold = 0.5

        # whether to completely disable traffic light faces in the semantic rasterizer
        self.rasterizer.disable_traffic_light_faces = False

        # When set to True, the rasterizer will set the raster origin at bottom left,
        # i.e. vehicles are driving on the right side of the road.
        # With this change, the vertical flipping on the raster used in the visualization code is no longer needed.
        # Set it to False for models trained before v1.1.0-25-g3c517f0 (December 2020).
        # In that case visualisation will be flipped (we've removed the flip there) but the model's input will be correct.
        self.rasterizer.set_origin_to_bottom = True

        #  if a tracked agent is closed than this value to ego, it will be controlled
        self.simulation.distance_th_close = 15
        self.simulation.distance_th_far = 50

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = True

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0


class L5KitVectorizedEnvConfig(EnvConfig):
    def __init__(self):
        super(L5KitVectorizedEnvConfig, self).__init__()
        self.name = "l5_vectorized"

        # the keys are relative to the dataset environment variable
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"
        self.rasterizer.dataset_meta_key = "meta.json"

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        self.rasterizer.filter_agents_threshold = 0.5

        # whether to completely disable traffic light faces in the semantic rasterizer
        # this disable option is not supported in avsw_semantic
        self.rasterizer.disable_traffic_light_faces = False

        self.data_generation_params.other_agents_num = 20
        self.data_generation_params.max_agents_distance = 50
        self.data_generation_params.lane_params.max_num_lanes = 15
        self.data_generation_params.lane_params.max_points_per_lane = MAX_POINTS_LANE
        self.data_generation_params.lane_params.max_points_per_crosswalk = 5
        self.data_generation_params.lane_params.max_retrieval_distance_m = 35
        self.data_generation_params.lane_params.max_num_crosswalks = 20

        #  if a tracked agent is closed than this value to ego, it will be controlled
        self.simulation.distance_th_far = 30

        #  if a new agent is closer than this value to ego, it will be controlled
        self.simulation.distance_th_close = 15

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = True

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0


class L5KitMixedEnvConfig(EnvConfig):
    """Vectorized Scene Component + Rasterized Map"""

    def __init__(self):
        super(L5KitMixedEnvConfig, self).__init__()
        self.name = "l5_mixed"

        # the keys are relative to the dataset environment variable
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"
        self.rasterizer.dataset_meta_key = "meta.json"

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        self.rasterizer.filter_agents_threshold = 0.5

        # whether to completely disable traffic light faces in the semantic rasterizer
        # this disable option is not supported in avsw_semantic
        self.rasterizer.disable_traffic_light_faces = False

        self.generate_agent_obs = False

        self.data_generation_params.other_agents_num = 20
        self.data_generation_params.max_agents_distance = 50
        self.data_generation_params.lane_params.max_num_lanes = 15
        self.data_generation_params.lane_params.max_points_per_lane = MAX_POINTS_LANE
        self.data_generation_params.lane_params.max_points_per_crosswalk = 5
        self.data_generation_params.lane_params.max_retrieval_distance_m = 35
        self.data_generation_params.lane_params.max_num_crosswalks = 20

        # step size of lane interpolation
        self.data_generation_params.lane_params.lane_interp_step_size = 5.0
        self.data_generation_params.vectorize_lane = True

        self.rasterizer.raster_size = (224, 224)

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = (0.5, 0.5)

        # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        self.rasterizer.ego_center = (0.25, 0.5)

        self.rasterizer.map_type = "semantic_debug"
        # self.rasterizer.map_type = "scene_semantic"

        # the keys are relative to the dataset environment variable
        self.rasterizer.satellite_map_key = "aerial_map/aerial_map.png"
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"

        # When set to True, the rasterizer will set the raster origin at bottom left,
        # i.e. vehicles are driving on the right side of the road.
        # With this change, the vertical flipping on the raster used in the visualization code is no longer needed.
        # Set it to False for models trained before v1.1.0-25-g3c517f0 (December 2020).
        # In that case visualisation will be flipped (we've removed the flip there) but the model's input will be correct.
        self.rasterizer.set_origin_to_bottom = True

        #  if a tracked agent is closed than this value to ego, it will be controlled
        self.simulation.distance_th_far = 50

        #  if a new agent is closer than this value to ego, it will be controlled
        self.simulation.distance_th_close = 50

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = False

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0


class L5KitMixedSemanticMapEnvConfig(L5KitMixedEnvConfig):
    def __init__(self):
        super(L5KitMixedSemanticMapEnvConfig, self).__init__()
        self.rasterizer.map_type = "py_semantic"


class L5RasterizedPlanningConfig(AlgoConfig):
    def __init__(self):
        super(L5RasterizedPlanningConfig, self).__init__()

        self.name = "l5_rasterized"
        self.model_architecture = "resnet50"
        self.map_feature_dim = 256
        self.history_num_frames = 5
        self.history_num_frames_ego = 5
        self.history_num_frames_agents = 5
        self.future_num_frames = 50
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()

        self.dynamics.type = None
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph
        self.dynamics.predict_current_states = False

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class SpatialPlannerConfig(L5RasterizedPlanningConfig):
    def __init__(self):
        super(SpatialPlannerConfig, self).__init__()
        self.name = "spatial_planner"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.loss_weights.pixel_res_loss = 1.0
        self.loss_weights.pixel_yaw_loss = 1.0


class MARasterizedPlanningConfig(L5RasterizedPlanningConfig):
    def __init__(self):
        super(MARasterizedPlanningConfig, self).__init__()
        self.name = "ma_rasterized"
        self.agent_feature_dim = 128
        self.context_size = (30, 30)


class L5RasterizedGCConfig(L5RasterizedPlanningConfig):
    def __init__(self):
        super(L5RasterizedGCConfig, self).__init__()
        self.name = "l5_rasterized_gc"
        self.goal_feature_dim = 32


class L5RasterizedVAEConfig(L5RasterizedPlanningConfig):
    def __init__(self):
        super(L5RasterizedVAEConfig, self).__init__()
        self.name = "l5_rasterized_vae"
        self.map_feature_dim = 256
        self.vae.latent_dim = 2
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)

        self.loss_weights.kl_loss = 1e-4


class L5TransformerPredConfig(AlgoConfig):
    def __init__(self):
        super(L5TransformerPredConfig, self).__init__()

        self.name = "TransformerPred"
        self.model_architecture = "Factorized"
        self.history_num_frames = 10
        self.history_num_frames_ego = 10  # this will also create raster history (we need to remove the raster from train/eval dataset - only visualization)
        self.history_num_frames_agents = 10
        self.future_num_frames = 20
        self.step_time = 0.2
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
        self.weights_scaling = [1.0, 1.0, 1.0]
        self.ego_weight = 1.0
        self.all_other_weight = 0.5
        self.disable_other_agents = False
        self.disable_map = False
        # self.disable_lane_boundaries = True
        self.global_head_dropout = 0.0
        self.training_num_N = 10000
        self.N_layer_enc = 2
        self.N_layer_tgt_enc = 1
        self.N_layer_tgt_dec = 1
        self.vmax = 30
        self.vmin = -10
        self.reg_weight = 10
        self.calc_likelihood = False
        self.calc_collision = False
        self.collision_weight = 1
        self.map_enc_mode = "all"
        self.temporal_bias = 0.5
        self.lane_regulation_weight = 1.5

        # map encoding parameters
        self.CNN.map_channels = 3
        self.CNN.hidden_channels = [10, 20, 20, 10]
        self.CNN.ROI_outdim = 10
        self.CNN.output_size = self.map_emb_dim
        self.CNN.patch_size = [15, 35, 25, 25]
        self.CNN.kernel_size = [5, 5, 5, 3]
        self.CNN.strides = [1, 1, 1, 1]
        self.CNN.input_size = [224, 224]
        self.CNN.veh_ROI_outdim = 4

        # Multi-modal prediction
        self.M = 1

        self.Discriminator.N_layer_enc = 1

        self.vectorize_map = False
        self.vectorize_lane = True
        self.points_per_lane = MAX_POINTS_LANE

        self.try_to_use_cuda = True

        # self.model_params.future_num_frames = 0
        # self.model_params.step_time = 0.2
        self.render_ego_history = False
        # self.model_params.history_num_frames_ego = 0
        # self.model_params.history_num_frames = 0
        # self.model_params.history_num_frames_agents = 0

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class L5TransformerGANConfig(L5TransformerPredConfig):
    def __init__(self):
        super(L5TransformerGANConfig, self).__init__()
        self.name = "TransformerGAN"
        self.calc_likelihood = True
        self.f_steps = 5
        self.GAN_weight = 0.2
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
