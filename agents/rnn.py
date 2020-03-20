import numpy as np 
import pandas as pd 
import tensorflow as tf 
from collections import deque
from poptim.agents.base import Agent 
from poptim.utils.numpy_utils import softmax 
from poptim.utils.preprocessor import rolling2d
from poptim.utils.pandas_utils import clean 

class RnnAgent(Agent):
    """
    Clase base para todos los agentes basados en redes recursivas.
    Predice el retorno futuro en base a observaciones anteriores.
    Luego, aplicando la función softmax sobre la predicción,
    decide el peso asignado a cada activo.
    """
    def __init__(self, action_space, df_training, window, hidden_units=50, policy='softmax', batch_size=32, epochs=400):
        
        self.action_space = action_space
        self.observation_size = self.action_size = len(df_training.columns)
        self.memory = deque(maxlen=window)
        self.model = self.build_model(hidden_units)
        X, y = self.split_sequences(df_training)
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
        self.policy = policy

    def build_model(self,hidden_units):
        raise NotImplementedError

    def observe(self, observation, action, reward, done, next_observation):
        self.memory.append(observation['returns'].values)

    def reshape_memory(self, memory):
        return memory.reshape((1, self.memory.maxlen, self.observation_size))

    def split_sequences(self, sequences):
        if isinstance(sequences, pd.DataFrame):
            sequences = sequences.values 
        X, y = [], []
        for i in range(len(sequences)):
            end_ix = i + self.memory.maxlen 
            if end_ix > len(sequences) - 1:
                break 
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def act(self, observation):

        memory = np.array(self.memory)

        if len(self.memory) != self.memory.maxlen:
            return self.action_space.sample()
        
        memory = self.reshape_memory(memory)
        prediction = self.model.predict(memory)

        if self.policy == 'softmax':
            action = pd.Series(prediction.ravel(), index=observation['returns'].index, name=observation['returns'].name)
            action = softmax(action)
            return action
        elif self.policy == 'best':
            action = np.zeros_like(prediction).ravel()
            action[np.argmax(prediction)] = 1.
            action = pd.Series(action, index=observation['returns'].index, name=observation['returns'].name)
            return action

class RnnLSTMAgent(RnnAgent):
    _id = 'rnn-lstm'

    def build_model(self, hidden_units):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(hidden_units, activation='relu', return_sequences=True, input_shape=(self.memory.maxlen, self.observation_size)))
        model.add(tf.keras.layers.LSTM(hidden_units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size))
        model.compile(optimizer='adam', loss='mse',verbose=0)
        return model 

class RnnGRUAgent(RnnAgent):
    _id = 'rnn-gru'

    def build_model(self, hidden_units):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(hidden_units, activation='relu', return_sequences=True, input_shape=(self.memory.maxlen, self.observation_size)))
        model.add(tf.keras.layers.GRU(hidden_units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size))
        model.compile(optimizer='adam', loss='mse',verbose=0)
        return model 

class RnnConvGRUAgent(RnnAgent):

    _id = 'rnn-conv-gru'

    def __init__(self, action_space, df_training, window, n_seq, hidden_units=50, policy='softmax', batch_size=32, epochs=400):
        
        self.action_space = action_space
        self.observation_size = self.action_size = len(df_training.columns)
        self.memory = deque(maxlen=window)
        self.n_seq = n_seq
        self.model = self.build_model(hidden_units)
        X, y = self.split_sequences(df_training)
        X = X.reshape((X.shape[0], self.n_seq, int(window/self.n_seq), len(df_training.columns)))
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
        self.policy = policy
        
    def build_model(self, hidden_units):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='causal', activation='relu'), input_shape=(None, int(self.memory.maxlen/self.n_seq), self.observation_size)))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        model.add(tf.keras.layers.GRU(hidden_units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.observation_size))
        model.compile(optimizer='adam', loss='mse',verbose=0)
        return model 

    def reshape_memory(self, memory):
        return memory.reshape((1, self.n_seq, int(self.memory.maxlen/self.n_seq), self.observation_size))