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
        self.policy.num_plan_samples = 10
        self.policy.num_action_samples = 10

        self.perturb.std = [5.0, 5.0, np.pi / 2]

        self.nusc.eval_scenes = [0, 1, 2, 3, 4]
        self.l5kit.eval_scenes = [9058, 5232, 14153, 8173, 10314, 7027, 9812, 1090, 9453, 978, 10263, 874, 5563, 9613, 261, 2826, 2175, 9977, 6423, 1069, 1836, 8198, 5034, 6016, 2525, 927, 3634, 11806, 4911, 6192, 11641, 461, 142, 15493, 4919, 8494, 14572, 2402, 308, 1952, 13287, 15614, 6529, 12, 11543, 4558, 489, 6876, 15279, 6095, 5877, 8928, 10599, 16150, 11296, 9382, 13352, 1794, 16122, 12429, 15321, 8614, 12447, 4502, 13235, 2919, 15893, 12960, 7043, 9278, 952, 4699, 768, 13146, 8827, 16212, 10777, 15885, 11319, 9417, 14092, 14873, 6740, 11847, 15331, 15639, 11361, 14784, 13448, 10124, 4872, 3567, 5543, 2214, 7624, 10193, 7297, 1308, 3951, 14001]
