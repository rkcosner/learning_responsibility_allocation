from tbsim.configs.l5kit_config import TrainConfig, AlgoConfig


class L5KitOnlineTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitOnlineTrainConfig, self).__init__()

        self.dataset_path = "YOUR_DAFA_FOLDER"
        self.dataset_valid_key = "scenes/validate.zarr"
        self.dataset_train_key = "scenes/train.zarr"
        self.dataset_meta_key = "meta.json"
        self.datamodule_class = "L5MixedDataModule"

        self.rollout.enabled = False  # disable external callback-based rollout

        self.train_rollout.every_n_steps = 100
        self.train_rollout.num_episodes = 10
        self.train_rollout.num_episodes_warm_start = 1
        self.train_rollout.num_scenes = 1
        self.train_rollout.n_step_action = 10
        self.train_rollout.horizon = 200

        self.test_rollout.every_n_steps = 500
        self.test_rollout.num_episodes = 10
        self.test_rollout.num_scenes = 3
        self.test_rollout.n_step_action = 10

        ## training config
        self.training.buffer_size = 100
        self.training.batch_size = 100
        self.training.num_steps = 200000
        self.training.num_data_workers = 4

        self.save.every_n_steps = 1000

        ## validation config
        self.validation.enabled = False
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 0.0


class SelfPlayHierarchicalConfig(AlgoConfig):
    def __init__(self):
        super(SelfPlayHierarchicalConfig, self).__init__()

        self.name = "sp_hierarchical"
        self.planner_ckpt_path = "checkpoints/spatial_archresnet50_bs64_pcl1.0_pbl0.0_rlFalse_2573128/iter55999_ep0_valCELoss2.68.ckpt"
        self.planner_config_path = "checkpoints/spatial_archresnet50_bs64_pcl1.0_pbl0.0_rlFalse_2573128/config.json"
        self.controller_ckpt_path = "checkpoints/gc_clip_regyaw_dynUnicycle_decmlp128,128_decstateTrue_yrl1.0_2596419/iter37999_ep0_valLoss0.06.ckpt"
        self.controller_config_path = "checkpoints/gc_clip_regyaw_dynUnicycle_decmlp128,128_decstateTrue_yrl1.0_2596419/config.json"

        self.history_num_frames = 5
        self.history_num_frames_ego = 5
        self.history_num_frames_agents = 5
        self.future_num_frames = 50
        self.step_time = 0.1
        self.render_ego_history = False

        self.policy.mask_drivable = True
        self.policy.sample = True