import numpy as np 
import tensorflow as tf 
from poptim.utils.reinforcement_learning import Replay_buffer, Prioritized_experience_replay, OrnsteinUhlenbeckProcess
import pickle 
from poptim.agents.base import Agent
import tqdm
from poptim.envs.trading import TradingEnv

class Actor:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim 
        self.action_dim = action_dim 

    def model(self):
        state = tf.keras.Input(shape=self.state_dim)
        x = tf.keras.layers.Dense(256, activation='relu')(state) 
        x = tf.keras.layers.GaussianNoise(1.)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.GaussianNoise(1.)(x)
        out = tf.keras.layers.Dense(self.action_dim, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform())(x)
        return tf.keras.Model(inputs=state, outputs=out)
    
class Critic:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim 
        self.action_dim = action_dim

    def model(self):
        state = tf.keras.Input(shape=self.state_dim, name='state_input')
        state_i = tf.keras.layers.Dense(256, activation='relu')(state)
        action = tf.keras.Input(shape=(self.action_dim,))
        x = tf.keras.layers.concatenate([state_i, action])
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        out = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform())(x)
        return tf.keras.Model(inputs=[state, action], outputs=out)

class DDPGAgent2(Agent):
    """
    Parámetros
    ==========

    actor: tf.keras.Model
        Red neuronal para el Actor.
    critic: tf.keras.Model
        Red neuronal para Critic.
    buffer: Replay_buffer | Prioritized_experience_replay
        Buffer pre grabado.
    max_buffer_size: int 
        Cantidad máxima de transiciones almacenadas.
    batch_size: int
        Batch size para entrenar las redes Actor y Critic.
    max_time_steps: int
        Número de pasos por época.
    tow: float
        Actualización suave de las redes objetivo (target).
    discount_factor: float
        Tasa de descuento de los rewards.
    explore_time: int
        Pasos temporales para acciones aleatorias (exploración).
    actor_learning_rate: float
        Tasa de aprendizaje del Actor.
    critic_learning_rate: float
        Tasa de aprendizaje del Critic.
    dtype: str
        Tipo de los datos.
    n_episodes: int 
        Número de episodios a correr.
    model_save_freq: int 
        Épocas para guardar el modelo y el buffer.
    """

    _id = 'ddpg-agent'

    def __init__(self, action_dim, state_dim, actor=None, critic=None, buffer=None, 
                 max_buffer_size=100000, batch_size=64, replay='uniform',
                tow=0.001, discount_factor=0.99, actor_learning_rate=0.0001,
                critic_learning_rate=0.0001, dtype='float64', n_episodes=500,
                verbose=False, model_save_freq=10):

        self.model_save_freq = model_save_freq
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        #self.T = max_time_steps 
        self.tow = tow 
        self.gamma = discount_factor
        #self.explore_time = int(0.1 * self.T)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.dflt_type = dtype
        self.n_episodes = n_episodes
        self.verbose = verbose 
        self.actor_opt = tf.keras.optimizers.Adam(self.actor_learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(self.critic_learning_rate)
        self.r, self.l, self.qlss = [], [], [] 
        self.observ_min = -np.inf ########### 
        self.observ_max = np.inf ############
        self.action_dim = action_dim ########
        self.state_dim = state_dim
        if buffer is not None:
            print('using loaded models')
            self.buffer = buffer 
            self.actor = actor 
            self.critic = critic 
        else:
            if replay=='prioritized':
                self.buffer = Prioritized_experience_replay(max_buffer_size, batch_size, dtype, self._id)
            else:
                self.buffer = Replay_buffer(max_buffer_size, batch_size, dtype, self._id)
            self.actor = Actor(state_dim, action_dim).model()
            self.critic = Critic(state_dim, action_dim).model()

        self.actor_target = Actor(state_dim, action_dim).model()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target = Critic(state_dim, action_dim).model()
        self.critic.compile(loss='mse', optimizer=self.critic_opt)
        self.critic_target.set_weights(self.critic.get_weights())
    
    def training_act(self, state):
        
        return self.actor.predict(state.reshape(1,-1))

    def act(self, state):
        
        w = self.actor.predict(state.reshape(1,-1))
        if np.any(w<0.):
            w = w + np.abs(np.min(w))
        w = w / np.sum(w)
        return w 

    def train_networks(self, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices=None):
        next_actions = self.actor_target(next_states_batch)
        next_actions = np.clip(next_actions, 0., 1.)
        next_actions = next_actions / np.sum(next_actions, axis=1).reshape(-1,1)
        q_t_pls_1 = self.critic_target([next_states_batch, next_actions])
        y_i = rewards_batch 
        for i in range(self.batch_size):
            if not done_batch[i]:
                y_i[i] += q_t_pls_1[i] * self.gamma 
        if isinstance(self.buffer, Prioritized_experience_replay):
            td_error = np.abs(y_i - self.critic.predict([states_batch, actions_batch]))
            self.buffer.update_priorities(indices, td_error)
        self.critic.train_on_batch([states_batch, actions_batch], y_i)

        with tf.GradientTape() as tape:
            a = self.actor(states_batch)
            tape.watch(a)
            q = self.critic([states_batch, a])
        dq_da = tape.gradient(q,a)

        with tf.GradientTape() as tape:
            a = self.actor(states_batch)
            theta = self.actor.trainable_variables 
        da_dtheta = tape.gradient(a, theta, output_gradients = -dq_da)
        self.actor_opt.apply_gradients(zip(da_dtheta,self.actor.trainable_variables))

    def update_target(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target

def train_ddpg(agent, mkt, df_prices, df_returns): 
    experience_cnt = 0
    ac = []
    rand=True 
    patience = 0
    for episode in tqdm.tqdm(range(agent.n_episodes)):
        #init_time = np.random.randint(int(df_returns.shape[0] * 0.5))
        #end_time = np.min([df_returns.shape[0], init_time + int(0.7 * df_returns.shape[0])])
        ri, li, qlssi = [], [], []
        env = TradingEnv(mkt=mkt, universe=None, prices=df_prices, returns=df_returns, cash=False)
        env.register(agent)
        T = env._max_episode_steps
        explore_time = int(0.35 * T)
        ob_t = env.reset()
        ob_t = np.array(ob_t['returns'])
        for t in range(T):
            action_t = agent.training_act(ob_t)
            if rand:
                noise = OrnsteinUhlenbeckProcess(size=agent.action_dim)
                action_t = action_t+noise.generate(t)
                if np.any(action_t<0.):
                    action_t = action_t + np.abs(np.min(action_t))
                action_t = action_t / np.sum(action_t)
            else:
                if np.any(action_t<0.):
                    action_t = action_t + np.abs(np.min(action_t))
                action_t = action_t / np.sum(action_t)
            ac.append(action_t)
            temp = env.step({agent.name: action_t.ravel()})
            ob_t_pls_1, rwrd_t, done_t = temp[0], temp[1], temp[2]
            ob_t_pls_1 = np.array(ob_t_pls_1['returns'])
            ri.append(rwrd_t[agent.name])
            agent.buffer.observe(ob_t.ravel(), action_t.ravel(), rwrd_t, ob_t_pls_1, done_t)
            ob_t = ob_t_pls_1

            if not rand:
                if isinstance(agent.buffer, Prioritized_experience_replay):
                    states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices = agent.buffer.sample_batch()
                    agent.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices)
                else:
                    states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = agent.buffer.sample_batch()
                    agent.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, None)

                agent.actor_target = agent.update_target(agent.actor_target, agent.actor, agent.tow)
                agent.critic_target = agent.update_target(agent.critic_target, agent.critic, agent.tow)

            if done_t or t == T - 1:
                rr = np.sum(ri)
                agent.r.append(rr)
                if agent.verbose:
                    print(f'Episode {episode} : Total Reward = {rr:.6f}')
                if len(agent.r) != 1 and agent.r[-2] == agent.r[-1]:
                    patience += 1                    
                break 
            if rand:
                experience_cnt += 1 
            if experience_cnt > explore_time:
                rand = False

        if agent.model_save_freq:
            if episode % agent.model_save_freq == 0:
                agent.actor.save('./ddpg/actor_model.h5')
                agent.critic.save('./ddpg/critic_model.h5')
                agent.actor_target.save('./ddpg/actor_target.h5')
                agent.critic_target.save('./ddpg/critic_target.h5')
                pickle.dump({'buffer':agent.buffer}, open('./ddpg/buffer.p','wb'))
        
        if patience == 100:
            break

