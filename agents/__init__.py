import poptim.agents.pretrainer
from poptim.agents.quadratic import QuadraticAgent
from poptim.agents.random import RandomAgent
from poptim.agents.uniform import UniformAgent
from poptim.agents.genetic import GeneticAgent
from poptim.agents.cla import CLAAgent
from poptim.agents.hrp import HRPAgent
from poptim.agents.aeagent import AEAgent, ShallowAEAgent, DeepAEAgent, RnnAEAgent
from poptim.agents.rnn import RnnAgent, RnnLSTMAgent, RnnGRUAgent, RnnConvGRUAgent
from poptim.agents.ddpg2 import DDPGAgent2, train_ddpg