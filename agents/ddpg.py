import numpy as np 
import tensorflow as tf 
from poptim.utils.memory_buffer import MemoryBuffer
import tqdm
from poptim.utils.reinforcement_learning import OrnsteinUhlenbeckProcess
from poptim.agents.base import Agent  

class Actor:

    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        
        self.env_dim = inp_dim 
        self.act_dim = out_dim 
        self.act_range = act_range 
        self.tau = tau 
        self.lr = lr 
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):

        inp = tf.keras.Input((self.env_dim))
        x = tf.keras.layers.Dense(256, activation='relu')(inp)
        x = tf.keras.layers.GaussianNoise(1.0)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.GaussianNoise(1.)(x)
        out = tf.keras.layers.Dense(self.act_dim, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform())(x)
        #out = tf.keras.layers.Lambda(lambda i: i * self.act_range)(out)
        return tf.keras.Model(inp, out)

    def predict(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self,inp):
        return self.target_model.predict(inp)

    def transfer_weights(self):

        w, target_w = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1. - self.tau)*target_w[i]
        self.target_model.set_weights(target_w)

    def train(self, states, actions, grads):
        self.adam_optimizer([states, grads])

    def optimizer(self):
        action_gdts = tf.keras.backend.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return tf.function([self.model.input, action_gdts], [tf.optimizers.Adam(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

class Critic:

    def __init__(self, inp_dim, out_dim, lr, tau):

        self.env_dim = inp_dim 
        self.act_dim = out_dim 
        self.tau, self.lr = tau, lr
        
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(tf.optimizers.Adam(self.lr), 'mse')
        self.action_grads = tf.function([self.model.input[0], self.model.input[1]], tf.gradients(self.model.output, [self.model.input[1]]))

    def network(self):

        state = tf.keras.Input((self.env_dim))
        action = tf.keras.Input((self.act_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(state)
        x = tf.keras.layers.concatenate([tf.keras.layers.Flatten()(x), action])
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        out = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform())(x)

        return tf.keras.Model([state, action], out)

    def gradients(self, states, actions):
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        return self.model.train_on_batch([states, actions], critic_target)
    
    def transfer_weights(self):
        w, target_w = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1. - self.tau) * target_w[i]
        self.target_model.set_weights(target_w)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
class DDPGAgent(Agent):

    _id = 'ddpg-agent'
    def __init__(self, action_space, df_training,nb_episodes=5000, k=4, batch_size=64,buffer_size=20000, gamma=0.99, lr=0.00005, tau=0.001):

        self.action_space = action_space
        self.nb_episodes = nb_episodes
        self.act_dim = len(df_training.columns)
        self.act_range = np.array([0.,1.])
        self.env_dim = (k, len(df_training.columns))
        self.gamma = gamma 
        self.lr = lr 
        self.actor = Actor(self.env_dim, self.act_dim, self.act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, self.act_dim, lr, tau)
        self.batch_size = batch_size
        self.buffer = MemoryBuffer(buffer_size)

    def act(self, s):
        action = self.actor.predict(s['returns'])[0]
        if np.any(action < 0.):
            action = action + np.abs(np.min(action))
        action = action / np.sum(action)
        return action
        #return self.actor.predict(s['returns'])[0]
    
    def bellman(self, rewards, q_values, dones):
        """Use the bellman equation to compute the critic target"""
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """Store experience in memory buffer"""

        self.buffer.memorize(state['returns'], action, reward, done, new_state['returns'])

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)
    
    def update_models(self, states, actions, critic_target):
        """Update actor and critic networks from sampled experience"""

        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        for i in range(len(actions)):
            if np.any(actions[i] < 0.):
                actions[i] = actions[i] + np.abs(np.min(actions[i]))
            actions[i] = actions[i] / np.sum(action)
        grads = self.critic.gradients(states, actions)
        #Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, env, args, summary_writer):
        results = []
        #tqdm_e = tqdm.tqdm(range(self.nb_episodes), desc='Score', leave=True, unit="episodes")
        for _ in range(self.nb_episodes):
            time, cumul_reward, done = 0, 0, False 
            old_state = env.reset()
            actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                a = self.act(old_state)
                a = a+noise.generate(time)
                if np.any(a<0.):
                    a = a + np.abs(np.min(a))
                a = a / np.sum(a)
                new_state, r, done, _ = env.step(a)
                self.memorize(old_state, a, r, done, new_state)
                states, actions, rewards, dones, new_states, _ = self.sample_batch(self.batch_size)
                q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                critic_target = self.bellman(rewards, q_values, dones)
                self.update_models(states, actions, critic_target)
                old_state = new_state
                cumul_reward += r 
                time += 1 
            #tqdm_e.set_description(f'Score: {cumul_reward}')
            #tqdm_e.refresh()
        
        return results 

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)