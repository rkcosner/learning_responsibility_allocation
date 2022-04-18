import torch
from typing import Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.l5_utils import get_drivable_region_map
from tbsim.utils.geometry_utils import calc_distance_map
from tbsim.utils.planning_utils import ego_sample_planning
from tbsim.policies.common import Action, Plan, RolloutAction


class HierarchicalWrapper(object):
    """A wrapper policy that feeds subgoal from a planner to a controller"""

    def __init__(self, planner, controller):
        self.device = planner.device
        self.planner = planner
        self.controller = controller

    def eval(self):
        self.planner.eval()
        self.controller.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        plan, plan_info = self.planner.get_plan(obs)
        actions, action_info = self.controller.get_action(
            obs,
            plan=plan,
            init_u=plan.controls
        )
        action_info["plan"] = plan.to_dict()
        plan_info.pop("plan_samples", None)
        action_info["plan_info"] = plan_info
        return actions, action_info


class HierarchicalSamplerWrapper(HierarchicalWrapper):
    """A wrapper policy that feeds plan samples from a stochastic planner to a controller"""

    def get_action(self, obs, **kwargs) -> Tuple[None, Dict]:
        _, plan_info = self.planner.get_plan(obs)
        plan_samples = plan_info.pop("plan_samples")
        b, n = plan_samples.positions.shape[:2]
        actions_tiled, _ = self.controller.get_action(
            obs,
            plan_samples=plan_samples,
            init_u=plan_samples.controls
        )

        action_samples = TensorUtils.reshape_dimensions(
            actions_tiled.to_dict(), begin_axis=0, end_axis=1, target_dims=(b, n)
        )
        action_samples = Action.from_dict(action_samples)

        action_info = dict(
            plan_samples=plan_samples,
            action_samples=action_samples,
            plan_info=plan_info
        )
        return None, action_info


class SamplingPolicyWrapper(object):
    def __init__(self, ego_action_sampler, agent_traj_predictor):
        """

        Args:
            ego_action_sampler: a policy that generates N action samples
            agent_traj_predictor: a model that predicts the motion of non-ego agents
        """
        self.device = ego_action_sampler.device
        self.sampler = ego_action_sampler
        self.predictor = agent_traj_predictor

    def eval(self):
        self.sampler.eval()
        self.predictor.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # actions of shape [B, num_samples, ...]
        _, action_info = self.sampler.get_action(obs)
        action_samples = action_info["action_samples"]
        agent_preds, _ = self.predictor.get_prediction(
            obs)  # preds of shape [B, A - 1, ...]

        ego_trajs = action_samples.trajectories
        agent_pred_trajs = agent_preds.trajectories

        agent_extents = obs["all_other_agents_future_extents"][..., :2].max(
            dim=-2)[0]
        drivable_map = get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        action_idx = ego_sample_planning(
            ego_trajectories=ego_trajs,
            agent_trajectories=agent_pred_trajs,
            ego_extents=obs["extent"][:, :2],
            agent_extents=agent_extents,
            raw_types=obs["all_other_agents_types"],
            raster_from_agent=obs["raster_from_agent"],
            dis_map=dis_map,
            weights={"collision_weight": 1.0, "lane_weight": 1.0},
        )

        ego_trajs_best = torch.gather(
            ego_trajs,
            dim=1,
            index=action_idx[:, None, None,
                  None].expand(-1, 1, *ego_trajs.shape[2:])
        ).squeeze(1)

        ego_actions = Action(
            positions=ego_trajs_best[..., :2], yaws=ego_trajs_best[..., 2:])
        action_info["action_samples"] = action_samples.to_dict()
        if "plan_samples" in action_info:
            action_info["plan_samples"] = action_info["plan_samples"].to_dict()
        return ego_actions, action_info


class PolicyWrapper(object):
    """A convenient wrapper for specifying run-time keyword arguments"""

    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def eval(self):
        self.model.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs, **kwargs)

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs, **kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)


class RolloutWrapper(object):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""

    def __init__(self, ego_policy=None, agents_policy=None,pass_agent_obs = True):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy
        self.pass_agent_obs = pass_agent_obs

    def eval(self):
        self.ego_policy.eval()
        self.agents_policy.eval()

    def get_action(self, obs, step_index) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            assert obs["ego"] is not None
            with torch.no_grad():
                if self.pass_agent_obs:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index,agent_obs = obs["agents"])
                else:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index)
        if self.agents_policy is not None:
            assert obs["agents"] is not None
            with torch.no_grad():
                agents_action, agents_action_info = self.agents_policy.get_action(
                    obs["agents"], step_index = step_index)
        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)