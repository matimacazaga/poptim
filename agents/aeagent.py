from collections import deque 
import numpy as np 
from poptim.agents.base import Agent 
import tensorflow as tf 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from scipy.stats import expon, uniform
from sklearn.preprocessing import MinMaxScaler

class StockPicking(BaseEstimator, TransformerMixin):
    """
    Clase para preprocesar los datos. Se seleccionan las n_top primeras y las ultimas
    n_bottom acciones del dataset market_information ordenado de manera ascendiente.

    Parámetros
    ==========
    market_information: pandas.DataFrame
        DataFrame con la norma l2 de la distancia entre los datos originales y 
        la reconstrucción realizada por un Autoencoder.
    n_top: int 
        Cantidad de acciones a considerar con menor norma l2.
    n_bottom: int 
        Cantidad de acciones a considerar con mayor norma l2.
    """

    def __init__(self, market_information, n_top=10, n_bottom=10):

        self.market_information = market_information 
        self.n_top = n_top 
        self.n_bottom = n_bottom 

    def fit(self, X, y=None):
        """
        Método para compatibilidad
        """
        
        return self

    def transform(self, X, y=None):
        """
        Método para transformar un dataset utilizando market_information. Ordena de
        manera ascendiente la información de mercado y selecciona las n_top
        acciones con menor l2-norm y las n_bottom con mayor. 

        Parámetros
        ==========
        X: numpy.array
            Array con retornos de las acciones.

        Retorno
        ======= 
        numpy.array
            Array con los retornos de las acciones seleccionadas.
        """

        sorted_cols = self.market_information.loc[:,'l2-norm'].argsort().values
        self.selected_stocks = np.concatenate([sorted_cols[:self.n_top], sorted_cols[-self.n_bottom:]])
        return X[:, self.selected_stocks]

class AEAgent(Agent):
    def __init__(self, action_space, min_neg_return, df_training, batch_size=32, window=10, max_n_top=20, max_n_bottom=20, *args):
        
        observation_space = df_training.shape[1]
        self.batch_size = batch_size
        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.memory_index = deque(maxlen=window)
        self.min_neg_return = min_neg_return
        self.df_training = df_training
        self.market_information = self.get_market_information(observation_space)
        self.max_n_top = max_n_top
        self.max_n_bottom = max_n_bottom
    
    def build_model(self, observation_space):
        raise NotImplementedError

    def observe(self, observation, action, reward, done, next_observation, index_ob):

        self.memory.append(observation['returns'].values)
        self.memory_index.append(index_ob)

    def reshape_training_data(self):
        return self.df_training.values

    def fit_model(self, model):
        model.fit(self.reshape_training_data(), self.reshape_training_data(), batch_size=32, epochs=400, shuffle=False, verbose=0)
    
    def make_prediction(self, model):
        return model.predict(self.reshape_training_data())

    def get_market_information(self, observation_space):

        ae = self.build_model(observation_space)
        self.fit_model(ae)
        reconstructed = pd.DataFrame(self.make_prediction(ae),
                                     columns=self.df_training.columns,
                                     index=self.df_training.index)

        market_information = np.linalg.norm(self.df_training.values - reconstructed.values, ord=None, axis=0)
        market_information = pd.DataFrame(market_information, index=reconstructed.columns,
                                          columns=['l2-norm'])
        return market_information

    def ammend(self, y):
        y_ammended = y 
        y_ammended[y_ammended < self.min_neg_return] = np.abs(self.min_neg_return)

    def act(self, observation):

        memory = np.array(self.memory)

        memory_index = np.array(self.memory_index)

        if len(self.memory) == self.memory.maxlen and len(self.memory_index) == self.memory_index.maxlen:

            pipeline_steps = [('transformer', StockPicking(self.market_information)),
                              ('model', ElasticNet(fit_intercept=False, max_iter=100000))]
        
            model = Pipeline(steps=pipeline_steps)

            params = {'transformer__n_top': range(1, self.max_n_top),
                  'transformer__n_bottom': range(1, self.max_n_bottom),
                  'model__alpha': uniform(),
                  'model__l1_ratio': uniform()}

            rand_search = RandomizedSearchCV(model, params, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1)

            memory = MinMaxScaler().fit_transform(memory)

            memory_index = MinMaxScaler().fit_transform(memory_index.reshape(-1,1))

            rand_search.fit(memory, memory_index)

            self.w = np.zeros(shape=self.market_information.shape[0])

            selected_columns = rand_search.best_estimator_['transformer'].selected_stocks

            self.w[selected_columns] = np.array(rand_search.best_estimator_['model'].coef_)

            if np.any(self.w<0.):
                self.w = self.w + np.abs(np.min(self.w))
            self.w = self.w / self.w.sum()

            return self.w
        else:
            return self.action_space.sample()

class ShallowAEAgent(AEAgent):

    _id = 'shallow-aeagent'

    def build_model(self, observation_space):
        encoder = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='relu', input_shape=[observation_space])])
        decoder = tf.keras.Sequential([tf.keras.layers.Dense(observation_space, input_shape=[5])])
        autoencoder = tf.keras.Sequential([encoder, decoder])
        autoencoder.compile(loss='mse', optimizer='adam')
        return autoencoder 

class DeepAEAgent(AEAgent):

    _id = 'deep-aeagent'

    def build_model(self, observation_space):
        stacked_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=[observation_space]),
            tf.keras.layers.Dense(30, activation='relu')
        ])
        stacked_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=[30]),
            tf.keras.layers.Dense(observation_space)
        ])
    
        stacked_autonecoder = tf.keras.Sequential([stacked_encoder, stacked_decoder])
        stacked_autonecoder.compile(loss='mse', optimizer='adam')
        return stacked_autonecoder        

class RnnAEAgent(AEAgent):

    _id = 'rnn-aeagent'

    def build_model(self, observation_space):
        encoder = tf.keras.Sequential([tf.keras.layers.LSTM(25, activation='relu', return_sequences=True, input_shape=(self.df_training.shape[0], observation_space)),
                                       tf.keras.layers.LSTM(10, activation='relu')])
        decoder = tf.keras.Sequential([tf.keras.layers.RepeatVector(self.df_training.shape[0], input_shape=[10]),
                                       tf.keras.layers.LSTM(25, return_sequences=True),
                                       tf.keras.layers.Dense(observation_space)])
        autoencoder = tf.keras.Sequential([encoder, decoder])
        autoencoder.compile(loss='mse', optimizer='adam')
        return autoencoder

    def reshape_training_data(self):
        return self.df_training.values.reshape((1, self.df_training.shape[0], self.df_training.shape[1]))

    def fit_model(self, model):
        model.fit(self.reshape_training_data(), self.reshape_training_data(), epochs=200, shuffle=False, verbose=0)
    
    def make_prediction(self, model):
        return model.predict(self.reshape_training_data())[0,:,:]