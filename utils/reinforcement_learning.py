import numpy as np 
from collections import deque 
import random 

class OrnsteinUhlenbeckProcess:

    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta 
        self.sigma = sigma 
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = -self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu 
        self.dt = dt
        self.size = size 

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0)*self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x 

class Replay_buffer:
    def __init__(self, max_buffer_size, batch_size, dflt_type, agent_name):
        self.batch_size = batch_size 
        self.buffer = deque(maxlen=max_buffer_size)
        self.dflt_type = dflt_type
        self.agent_name = agent_name

    def observe(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample_batch(self):
        replay_buffer = np.array(random.sample(self.buffer, self.batch_size))
        states_batch = np.array([i[0] for i in replay_buffer])
        actions_batch = np.array([i[1] for i in replay_buffer])
        rewards_batch = np.array([i[2][self.agent_name] for i in replay_buffer])
        next_states_batch = np.array([i[3] for i in replay_buffer])
        done_batch = np.array([i[4] for i in replay_buffer])
        #arr = np.array(replay_buffer)

        #states_batch = np.vstack(arr[:,0])
        #actions_batch = arr[:,1].astype(self.dflt_type).reshape(-1,1)
        #rewards_batch = arr[:, 2].astype(self.dflt_type).reshape(-1,1)
        #next_states_batch = np.vstack(arr[:,3])
        #done_batch = np.vstack(arr[:,4]).astype(bool)
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch 

class Prioritized_experience_replay:
    def __init__(self, max_buffer_size, batch_size, dflt_type, agent_name):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.priorities = deque(maxlen=max_buffer_size)
        self.indexes = deque(maxlen=max_buffer_size)
        self.dflt_type = dflt_type
        self.agent_name = agent_name 

    def observe(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
        self.priorities.append(1)
        ln = len(self.buffer)
        if ln < self.max_buffer_size:
            self.indexes.append(ln)
    
    def update_priorities(self,indices,priorities):
        for indx, priority in zip(indices,priorities):
            self.priorities[indx-1] = priority + 1

    def sample_batch(self):
        indices = random.choices(self.indexes, weights=self.priorities,k=self.batch_size)
        replay_buffer = [self.buffer[indx-1] for indx in indices]
        states_batch = np.array([i[0] for i in replay_buffer])
        actions_batch = np.array([i[1] for i in replay_buffer])
        rewards_batch = np.array([i[2][self.agent_name] for i in replay_buffer])
        next_states_batch = np.array([i[3] for i in replay_buffer])
        done_batch = np.array([i[4] for i in replay_buffer])
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices 
