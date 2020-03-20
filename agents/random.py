import numpy as np 
from poptim.agents.base import Agent

class RandomAgent(Agent):
    """
    Agente aleatorio.
    """

    _id = 'random'

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()