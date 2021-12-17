from tbsim.configs.config import Config


class TrainConfig(Config):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.logging.terminal_output_to_txt = True       # whether to log stdout to txt file
        self.logging.log_tb = True                       # enable tensorboard logging
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100

        ## save config - if and when to save model checkpoints ##
        self.save.enabled = True                         # whether model saving should be enabled or disabled
        self.save.every_n_steps = 100                     # save model every n epochs
        self.save.best_k = 5

        ## rendering config ##
        self.render.on_screen = False                    # render on-screen or not
        self.render.to_video = True                      # render evaluation rollouts to videos

        ## evaluation rollout config ##
        self.rollout.enabled = False                     # enable evaluation rollouts
        self.rollout.num_episodes = 10                   # number of evaluation episodes to run
        self.rollout.num_scenes = 25                     # number of parallel scenes to run (if applicable)
        self.rollout.every_n_steps = 1000                # do rollouts every @rate epochs
        self.rollout.warm_start_n_steps = 1              # number of steps to wait before starting rollouts

        ## training config
        self.training.batch_size = 100
        self.training.num_steps = 200000
        self.training.num_data_workers = 0

        ## validation config
        self.validation.enabled = False
        self.validation.batch_size = 100
        self.validation.num_data_workers = 0
        self.validation.every_n_steps = 1000
        self.validation.num_steps_per_epoch = 100


class EnvConfig(Config):
    def __init__(self):
        super(EnvConfig, self).__init__()
        self.name = "my_env"


class AlgoConfig(Config):
    def __init__(self):
        super(AlgoConfig, self).__init__()
        self.name = "my_algo"


class ExperimentConfig(Config):
    def __init__(
            self,
            train_config: TrainConfig,
            env_config: EnvConfig,
            algo_config: AlgoConfig,
            registered_name: str = None
    ):
        """

        Args:
            train_config (TrainConfig): training config
            env_config (EnvConfig): environment config
            algo_config (AlgoConfig): algorithm config
            registered_name (str): name of the experiment config object in the global config registry
        """
        super(ExperimentConfig, self).__init__()
        self.registered_name = registered_name

        # Write all results to this directory. A new folder with the timestamp will be created
        # in this directory, and it will contain three subfolders - "log", "models", and "videos".
        # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
        # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
        # videos.
        self.root_dir = "{}_trained_models/".format(self.algo.name)
        self.name = "test"                # name of the experiment (creates a subdirectory under root_dir)
        self.seed = 1                     # seed for everything (for reproducibility)

        self.devices.num_gpus = 1         # Set to 0 to use CPU

        self.train = train_config
        self.env = env_config
        self.algo = algo_config
