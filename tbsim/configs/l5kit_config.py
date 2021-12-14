from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig


class L5KitTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitTrainConfig, self).__init__()

        self.dataset_path = "/home/danfeix/workspace/lfs/lyft/lyft_prediction/"
        self.dataset_valid_key = "scenes/validate.zarr"
        self.dataset_train_key = "scenes/train.zarr"
        self.dataset_meta_key = "meta.json"

        self.training.num_data_workers = 2
        self.validation.num_data_workers = 2
        self.validation.enabled = True
        self.rollout.enabled = True
        self.rollout.every_n_steps = 1000
        self.rollout.num_episodes = 30
        self.save.every_n_steps = 1000


class L5KitEnvConfig(EnvConfig):
    def __init__(self):
        super(L5KitEnvConfig, self).__init__()

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


class L5RasterizedPlanningConfig(AlgoConfig):
    def __init__(self):
        super(L5RasterizedPlanningConfig, self).__init__()

        self.name = "l5_rasterized"
        self.model_architecture = "resnet50"
        self.history_num_frames = 5
        self.future_num_frames = 50
        self.step_time = 0.1
        self.render_ego_history = False

        self.optim_params.policy.learning_rate.initial = 1e-3      # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength
