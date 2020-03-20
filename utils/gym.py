import numpy as np 
from gym import Env 

import typing 
import os 

def run(env: Env, agent,
        num_episodes: int, record: bool = True,
        log: bool = False):

    if hasattr(env, 'unregister'):
        env.unregister(agent=None)

    if hasattr(env, 'register'):

        if not hasattr(agent, 'name'):
            agent.name = '_default'

        env.register(agent)

    rewards = []

    actions = []

    _best_reward = -np.inf 

    def _run() -> typing.Tuple[typing.List[float], typing.List[np.ndarray]]:

        _rewards = []

        _actions = []

        ob = env.reset()

        reward = 0.0 

        done = False 

        info = {}

        j = 0

        agent.begin_episode(ob)

        while (not done) and (j < env._max_episode_steps):

            action = agent.act(ob)

            if hasattr(env, 'register'):

                ob_, reward, done, info = env.step({agent.name: action})

                reward = reward[agent.name]
            
            else:

                ob_, reward, done, info = env.step(action)

            _rewards.append(reward)

            _actions.append(action)

            agent.observe(ob, action, reward, done, ob_)

            ob = ob_ 

            j += 1

        agent.end_episode()
        
        return _rewards, _actions 

    for e in range(num_episodes):

        R, A = _run()

        if record:

            rewards.append(R)

            actions.append(A)

        if log:
            print(f'episode {e:4d}, cumulative reward: {sum(R):.5f}')
        
        if sum(R) > _best_reward:

            try:

                os.remove(f'tmp/models/{agent.name}/{_best_reward}.h5')
            except:
                pass
            
            _best_reward = sum(R)

            if hasattr(agent, 'save'):

                if not os.path.exists(f'tmp/models/{agent.name}'):
                    os.makedirs(f'tmp/models/{agent.name}')
                
                agent.save(f'tmp/models/{agent.name}/{_best_reward}.h5')

    return rewards, actions 
