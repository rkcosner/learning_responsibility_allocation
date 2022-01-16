import torch
import numpy as np
import math, copy, time
import abc
from copy import deepcopy


class DynType:
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """

    UNICYCLE = 1
    SI = 2
    DI = 3


class Dynamics(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name, **kwargs):
        self.xdim = 4
        self.udim = 2

    @abc.abstractmethod
    def __call__(self, x, u):
        return

    @abc.abstractmethod
    def step(self, x, u, dt, bound=True):
        return

    @abc.abstractmethod
    def name(self):
        return

    @abc.abstractmethod
    def type(self):
        return

    @abc.abstractmethod
    def ubound(self, x):
        return

    @staticmethod
    def state2pos(x):
        return

    @staticmethod
    def state2yaw(x):
        return