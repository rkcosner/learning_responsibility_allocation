import math

from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig

MAX_POINTS_LANE = 5


class NuscTrainConfig(TrainConfig):
    def __init__(self):
        super(NuscTrainConfig, self).__init__()

        self.avdata_source = "nusc_mini"  # [nusc_mini, nusc, lyft_sample, lyft]
        self.dataset_path = "SET-THIS-THROUGH-TRAIN-SCRIPT-ARGS"
        self.datamodule_class = "UnifiedDataModule"

        self.rollout.enabled = False
        self.rollout.every_n_steps = 500
        self.rollout.num_episodes = 10
        self.rollout.num_scenes = 3
        self.rollout.n_step_action = 10

        # training config
        self.training.batch_size = 100
        self.training.num_steps = 200000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 1000
        self.save.best_k = 10

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 6
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 50


class NuscEnvConfig(EnvConfig):
    def __init__(self):
        super(NuscEnvConfig, self).__init__()

        self.name = "nusc_avdata"

        # raster image size [pixels]
        self.rasterizer.raster_size = 224

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 0.5

        # maximum number of agents to consider during training
        self.data_generation_params.other_agents_num = 20

        self.data_generation_params.max_agents_distance = 30

        self.simulation.distance_th_close = 30

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = False

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0