from collections import deque 

import numpy as np 
import scipy.optimize as sp_opt 

from poptim.agents.pretrainer.optimizer import optimizer
from poptim.agents.pretrainer.objectives import sharpe_ratio, risk_aversion
from poptim.agents.base import Agent

class QuadraticAgent(Agent):

    _id = 'quadratic'

    def __init__(self, action_space, J, window=10, *args):
        
        self.optimizer = optimizer(self._J(J), *args)
        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()

    def observe(self, observation, action, reward, done, next_observation):

        self.memory.append(observation.values)

    def act(self, observation):
        
        # Para simplificar
        memory = np.array(self.memory)

        # Numero de acciones
        M = len(observation)

        # Vector de retornos esperados
        mu = np.mean(memory, axis=0).reshape(M,1)

        if len(self.memory) != self.memory.maxlen:

            sigma = np.eye(M)

        else:

            sigma = np.cov(memory.T)

        try:
            self.w = self.optimizer(mu, sigma, self.w)

        except BaseException:
            pass

        return self.w 

    def _J(self, J):

        if J is 'sharpe_ratio':
            return sharpe_ratio

        elif J is 'risk_aversion':
            return risk_aversion

    
