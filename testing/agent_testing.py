import poptim
import gym
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
import tensorflow as tf
import tqdm
sns.set()

import multiprocessing
from joblib import Parallel, delayed

# agents
#from poptim.agents.random import RandomAgent
#from poptim.agents.uniform import UniformAgent
#from poptim.agents.hrp import HRPAgent
from poptim.agents.cla import CLAAgent
#from poptim.agents.aeagent import ShallowAEAgent, DeepAEAgent, RnnAEAgent
#from poptim.agents.rnn import RnnGRUAgent, RnnLSTMAgent, RnnConvGRUAgent

np.random.seed(42)
tf.random.set_seed(42)

WINDOW = 50
N_SEQ = 10

path='C:/Users/matim/Documents/TesisMaestria/'

config = pickle.load(open(f'{path}config.p', 'rb'))

prices = pickle.load(open(f'{path}prices_usa.p', 'rb'))
returns = prices.pct_change().dropna()

prices_training = prices.loc[config['Init Training']:config['End Training'],:]
prices_backtesting = prices.loc[config['Init Backtesting']:config['End Backtesting'], :]
returns_training = returns.loc[config['Init Training']:config['End Training'],:]
returns_backtesting = returns.loc[config['Init Backtesting']:config['End Backtesting'], :]

gspc_prices = pd.read_csv(f'{path}gspc.csv', index_col=0, parse_dates=True)
gspc_returns = gspc_prices.pct_change().dropna().loc[:, 'Adj Close']

gspc_prices_training = gspc_prices.loc[config['Init Training']:config['End Training']]
gspc_prices_backtesting = gspc_prices.loc[config['Init Backtesting']:config['End Backtesting']]

gspc_returns_training = gspc_returns.loc[config['Init Training']:config['End Training']]
gspc_returns_backtesting = gspc_returns.loc[config['Init Backtesting']:config['End Backtesting']]

MIN_NEG_RETURN = gspc_returns_training[gspc_returns_training < 0.].mean()

random_universes = pickle.load(open(f'{path}random_universes.p','rb'))

def sim_cla_mv(i,universe):
    env = poptim.envs.TradingEnv(mkt='USA', universe=None, prices=prices_backtesting.loc[:,universe], returns=returns_backtesting.loc[:,universe], cash=False)
    agent = poptim.agents.cla.CLAAgent(action_space=env.action_space, J='min variance', window=WINDOW)
    env.register(agent)
    
    ob = env.reset()
    ob = ob['returns']
    reward = 0 
    done = False 
    for _ in range(env._max_episode_steps):
        if done:
            break
        agent.observe(ob, None, reward, done, None)
        action = agent.act(ob)
        ob, reward, done, _ = env.step({agent.name: action})
        ob = ob['returns']  
    
    pickle.dump(env.agents[agent.name], open(f'./bootstrapping_backtesting/{agent.name}_{i}.p','wb'))   


num_cores = multiprocessing.cpu_count()
print(num_cores)
inputs = tqdm.tqdm([(i, ri) for i,ri in zip(range(73,100),random_universes[73:])])
Parallel(n_jobs=num_cores)(delayed(sim_cla_mv)(i,universe) for i,universe in inputs)