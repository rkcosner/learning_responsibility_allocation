from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig


class L5KitTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitTrainConfig, self).__init__()

        self.dataset_path = "YOUR-DATASET-PATH"
        self.dataset_valid_key = "scenes/validate.zarr"
        self.dataset_train_key = "scenes/train.zarr"
        self.dataset_meta_key = "meta.json"
        self.datamodule_class = "L5RasterizedDataModule"

        self.rollout.enabled = True
        self.rollout.every_n_steps = 1000
        self.rollout.num_episodes = 2
        self.rollout.num_scenes = 25

        ## training config
        self.training.batch_size = 100
        self.training.num_steps = 2000000
        self.training.num_data_workers = 4

        self.save.every_n_steps = 2000

        ## validation config
        self.validation.enabled = True
        self.validation.batch_size = 100
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 1000
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
        self.simulation.distance_th_far = 30

        #  if a new agent is closer than this value to ego, it will be controlled
        self.simulation.distance_th_close = 15

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = True

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
        self.data_generation_params.lane_params.max_points_per_lane = 5
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

        self.data_generation_params.other_agents_num = 20
        self.data_generation_params.max_agents_distance = 50
        self.data_generation_params.lane_params.max_num_lanes = 15
        self.data_generation_params.lane_params.max_points_per_lane = 5
        self.data_generation_params.lane_params.max_points_per_crosswalk = 5
        self.data_generation_params.lane_params.max_retrieval_distance_m = 35
        self.data_generation_params.lane_params.max_num_crosswalks = 20

        self.rasterizer.raster_size = (224, 224)

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = (0.5, 0.5)

        # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        self.rasterizer.ego_center = (0.25, 0.5)

        self.rasterizer.map_type = "semantic_debug"

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
        self.simulation.distance_th_far = 30

        #  if a new agent is closer than this value to ego, it will be controlled
        self.simulation.distance_th_close = 15

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = True

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0


class L5RasterizedPlanningConfig(AlgoConfig):
    def __init__(self):
        super(L5RasterizedPlanningConfig, self).__init__()

        self.name = "l5_rasterized"
        self.model_architecture = "resnet50"
        self.history_num_frames = 5
        self.future_num_frames = 50
        self.step_time = 0.1
        self.render_ego_history = False
        self.dynamics.type = None
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = 8.0
        self.dynamics.acce_bound = (-6, 4)

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class L5RasterizedVAEConfig(L5RasterizedPlanningConfig):
    def __init__(self):
        super(L5RasterizedVAEConfig, self).__init__()
        self.name = "l5_rasterized_vae"
        self.visual_feature_dim = 256
        self.vae.latent_dim = 16
        self.vae.condition_dim = 16
        self.vae.kl_weight = 1e-4
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)


class L5TransformerPredConfig(AlgoConfig):
    def __init__(self):
        super(L5TransformerPredConfig, self).__init__()

        self.name = "TransformerPred"
        self.model_architecture = "Factorized"
        self.history_num_frames = 8
        self.history_num_frames_ego = 8  # this will also create raster history (we need to remove the raster from train/eval dataset - only visualization)
        self.history_num_frames_agents = 8
        self.future_num_frames = 10
        self.step_time = 0.2
        self.N_t = 4
        self.N_a = 3
        self.d_model = 512
        self.d_ff = 2048
        self.head = 8
        self.dropout = 0.1
        self.XY_step_size = [0.1, 0.1]
        self.weights_scaling = [1.0, 1.0, 1.0]
        self.ego_weight = 1.0
        self.all_other_weight = 0.5
        self.disable_other_agents = False
        self.disable_map = False
        # self.disable_lane_boundaries = True
        self.global_head_dropout = 0.0

        # map encoding parameters
        self.map_channels = 3
        self.hidden_channels = [10, 20, 10, 3]
        self.ROI_outdim = 12
        self.output_size = self.d_model
        self.patch_size = [15, 35, 25, 25]
        self.kernel_size = [5, 5, 5, 3]
        self.strides = [1, 1, 1, 1]
        self.input_size = [224, 224]

        self.try_to_use_cuda = True

        # self.model_params.future_num_frames = 0
        # self.model_params.step_time = 0.1
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
