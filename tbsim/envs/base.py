import abc


class SimulationException(Exception):
    pass


class BaseEnv(abc.ABC):
    """TODO: Make a Simulator MetaClass"""

    @abc.abstractmethod
    def reset(self):
        return

    @abc.abstractmethod
    def step(self, action, num_steps_to_take):
        return

    @abc.abstractmethod
    def get_metrics(self):
        return

    @abc.abstractmethod
    def render(self, actions_to_take):
        return

    @abc.abstractmethod
    def get_info(self):
        return

    @abc.abstractmethod
    def get_observation(self):
        return

    @abc.abstractmethod
    def get_reward(self):
        return

    @abc.abstractmethod
    def is_done(self):
        return

    @abc.abstractmethod
    def get_info(self):
        return


class BatchedEnv(abc.ABC):
    @abc.abstractmethod
    def num_instances(self):
        return
