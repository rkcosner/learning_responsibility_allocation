from robomimic.config import Config


class TrainConfig(Config):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.logging.terminal_output_to_txt = True       # whether to log stdout to txt file
        self.logging.log_tb = True                       # enable tensorboard logging

        # Write all results to this directory. A new folder with the timestamp will be created
        # in this directory, and it will contain three subfolders - "log", "models", and "videos".
        # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
        # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
        # videos.

        ## save config - if and when to save model checkpoints ##
        self.save.enabled = True                         # whether model saving should be enabled or disabled
        self.save.every_n_seconds = None                 # save model every n seconds (set to None to disable)
        self.save.every_n_epochs = 50                    # save model every n epochs (set to None to disable)
        self.save.epochs = []                            # save model on these specific epochs
        self.save.on_best_validation = False             # save models that achieve best validation score
        self.save.on_best_rollout_return = False         # save models that achieve best rollout return
        self.save.on_best_rollout_success_rate = True    # save models that achieve best success rate


        ## rendering config ##
        self.render = False                              # render on-screen or not
        self.render_video = True                         # render evaluation rollouts to videos
        self.keep_all_videos = False                     # save all videos, instead of only saving those for saved model checkpoints
        self.video_skip = 5                              # render video frame every n environment steps during rollout

        ## evaluation rollout config ##
        self.rollout.enabled = False                     # enable evaluation rollouts
        self.rollout.n = 50                              # number of rollouts per evaluation
        self.rollout.horizon = 400                       # maximum number of env steps per rollout
        self.rollout.rate = 50                           # do rollouts every @rate epochs
        self.rollout.warmstart = 0                       # number of epochs to wait before starting rollouts
        self.rollout.terminate_on_success = True         # end rollout early after task success

        ## training config
        self.training.num_data_workers = 2
        self.training.batch_size = 100
        self.training.num_epochs = 2000
        self.training.epoch_every_n_steps = 10          # number of gradient steps in train epoch (None for full dataset pass)

        ## validation config
        self.validation.num_data_workers = 0
        self.validation.enabled = True
        self.validation.epoch_every_n_steps = 10
        self.validation.batch_size = 100

        ## learning config ##
        self.cuda = True          # use GPU or not

        self.seed = 1             # seed for training (for reproducibility)


class EnvConfig(Config):
    def __init__(self):
        super(EnvConfig, self).__init__()
        self.name = "my_env"


class AlgoConfig(Config):
    def __init__(self):
        super(AlgoConfig, self).__init__()
        self.name = "my_algo"


class ExperimentConfig(Config):
    def __init__(self, train_config: TrainConfig, env_config: EnvConfig, algo_config: AlgoConfig):
        """

        Args:
            train_config (TrainConfig): training config
            env_config (EnvConfig): environment config
            algo_config (AlgoConfig): algorithm config
        """
        super(ExperimentConfig, self).__init__()
        self.train = train_config
        self.env = env_config
        self.algo = algo_config

        self.name = "test"
        self.root_dir = "../{}_trained_models".format(self.algo.name)
