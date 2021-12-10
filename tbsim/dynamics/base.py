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


class dynamic(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, x, u):
        return

    @abc.abstractmethod
    def step(self, x, u, dt):
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

    @abc.abstractmethod
    def state2pos(x):
        return
