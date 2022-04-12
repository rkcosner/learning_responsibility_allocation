import numpy as np
from tbsim.configs.config import Dict


class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = None
        self.env = "nusc"  # [l5kit, nusc]
        self.dataset_path = None
        self.eval_class = "HierAgentAware"
        self.seed = 0
        self.parallel_simulation = False
        self.num_parallel = 3
        self.num_scenes_per_batch = 5
        self.num_scenes_to_evaluate = 100
        self.num_simulation_steps = 200
        self.ego_only = False
        self.n_step_action = 5
        self.ckpt_dir = "checkpoints/"
        self.experience_hdf5_path = None
        self.results_dir = "results/"
        self.skip_first_n = 1

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 50

        self.perturb.std = [5.0, 5.0, np.pi / 2]