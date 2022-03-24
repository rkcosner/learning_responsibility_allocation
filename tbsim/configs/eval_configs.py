from tbsim.configs.config import Dict


class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = "eval"
        self.dataset_path = None
        self.eval_class = "HierAgentAware"
        self.seed = 0
        self.num_scenes_per_batch = 4
        self.num_scenes_to_evaluate = 100
        self.ego_only = False
        self.n_step_action = 5
        self.ckpt_dir = "checkpoints/"

        self.render_to_video = False
        self.results_dir = "results/"

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 10