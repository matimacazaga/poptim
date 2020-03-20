import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import typing 
import gym 
from poptim.agents.base import Agent 
from abc import abstractmethod
import poptim.utils 
import poptim.envs

class BaseEnv(gym.Env):
    """
    Trading Environment basado en OpenAI gym.

    Atributos
    =========

    universe: list
        Lista de activos.
    action_space: envs.spaces.PortfolioVector
        Espacio de acciones del agente.
    observation_space: gym.Space
        Espacio de observación del agente.
    prices: pandas.DataFrame
        Precios históricos para el 'universe'.
    returns: pandas.DataFrame
        Retornos históricos (porcentuales) para el 'universe'.   
    agents: list 
        Agentes registrados que compiten en el environment.
    
    Métodos
    =======

    step(action)
        El agente da un step en el environment.
    reset()
        Resetea el estado del environment y devuelve
        una observación inicial.
    render()
        Presenta data en tiempo real en un dashboard.
    register(agent)
        Añade un agente al environment.
    """

    class Record:
        """
        Estructura local para registros de acciones
        y recompensas.

        Atributos
        =========

        actions: pandas.DataFrame
            Tabla de acciones tomadas por el agente.
        rewards: pandas.DataFrame
            Tabla de las recompensas recibidas por el 
            agente. 
        """

        def __init__(self, index, columns):

            self.actions = pd.DataFrame(columns=columns,
                                        index=index,
                                        dtype=float)

            self.actions.iloc[0] = np.zeros(len(columns))
            self.actions.iloc[0]['Cash'] = 1.0
            self.rewards = pd.DataFrame(columns=columns,
                                        index=index,
                                        dtype=float)
            self.rewards.iloc[0] = np.zeros(len(columns))

    def __init__(self,
                 mkt: str, 
                 universe: typing.Optional[typing.List[str]]=None, 
                 prices: typing.Optional[pd.DataFrame]=None,
                 returns: typing.Optional[pd.DataFrame]=None,
                 cash: bool=True,
                 **kwargs):
        self.mkt = mkt 
        if (universe is None) and (prices is None):
            raise ValueError('O bien "universe" o "prices" debe ser Non-None')

        if prices is not None and isinstance(prices, pd.DataFrame):
            self._prices = poptim.utils.pandas_utils.clean(prices)
        elif universe is not None and isinstance(universe, list):
            self._prices = poptim.utils.pandas_utils.clean(self._get_prices(universe, **kwargs))

        if cash:
            self._prices['CASH'] = 1.0 
        
        if returns is not None and isinstance(prices, pd.DataFrame):
            self._returns = poptim.utils.pandas_utils.clean(returns)
        elif returns is None and (self._prices is not None and isinstance(self._prices, pd.DataFrame)):
            self._returns = self._prices.pct_change().dropna()

        num_instruments: int = len(self.universe)

        self.action_space = poptim.envs.spaces.PortfolioVector(num_instruments)
        self.observación = gym.spaces.Box(-np.inf,
                                          np.inf,
                                          (num_instruments,),
                                          dtype=np.float32)
        
        # Contador para seguir el índice temporal
        self._counter = 0

        self.agents = {}

        self._pnl = pd.DataFrame(index=self.dates,
                                 columns=[agent.name for agent in self.agents])
        
        self._fig, self._axes = None, None

    @property
    def universe(self):
        """Lista de activos."""
        return self._prices.columns.tolist()

    @property
    def dates(self):
        """Fechas de los precios del environment."""
        return self._prices.index 
    
    @property
    def index(self) -> pd.DatetimeIndex:
        """Indice actual."""
        return self.dates[self._counter]

    @property
    def _max_episode_steps(self) -> int:
        """Número de pasos temporales disponibles."""
        return len(self.dates)

    @abstractmethod
    def _get_prices(self, universe, trading_period, **kwargs) -> pd.DataFrame:
        raise NotImplementedError 

    def _get_observation(self) -> object:
        ob = {}
        ob['prices'] = self._prices.loc[self.index, :]
        ob['returns'] = self._returns.loc[self.index, :]
        return ob 
    
    def _get_reward(self, action) -> pd.Series:
        return self._returns.loc[self.index] * action 

    def _get_done(self) -> bool:
        return self.index == self.dates[-1]

    def _get_info(self) -> dict:
        return {}

    def _validate_agents(self):
        """Corrobora la disponibilidad de agentes."""
        if len(self.agents) == 0:
            raise RuntimeError('No existen agentes registrados en el environment')

    def register(self, agent:Agent):
        """Registra un agente en el environment."""
        if not hasattr(agent,'name'):
            raise ValueError('El agente debe tener un atributo "name".')
        if agent.name not in self.agents:
            self.agents[agent.name] = self.Record(columns=self.universe,
                                                  index=self.dates)
    
    def unregister(self, agent:typing.Optional[Agent]):
        """Eliminar un agente del environment. Si agent=None
           se eliminan todos los agentes."""

        if agent is None:
            self.agents = {}
            return None 

        if not hasattr(agent,'name'):
            raise ValueError('El agente debe tener un atributo "name".')

        if agent.name in self.agents:
            del self.agents[agent.name]

    def step(self, action:typing.Union[object, typing.Dict[str, object]]):
        """
        El agente toma un step en el environment.

        Parámetros
        ==========

        action: numpy.array | dict
            Portfolio vector(s)

        Retorno
        =======

        observation, reward, done, info: tuple
            * observation: object
                Observación del environment
            * reward: float | dict 
                Recompensa(s) recibidas luego del step.
            * done: bool
                Marca de episodio terminado
            * info: dict
                Información acerca del step
        """

        self._validate_agents()

        self._counter +=1

        observation = self._get_observation()
        done = self._get_done()
        info = self._get_info()

        if action.keys() != self.agents.keys():
            raise ValueError('Interfaz de acciones inválida')

        reward = {}

        for name, A in action.items():
            if not self.action_space.contains(A):
                raise ValueError(f'Acción intentada inválida {A}')

            self.agents[name].actions.loc[self.index] = A
            self.agents[name].rewards.loc[self.index] = self._get_reward(A)
            reward[name] = self.agents[name].rewards.loc[self.index].sum()

        return observation, reward, done, info 

    def reset(self) -> object:
        """ 
        Resetea el estado del environment y devuelve una
        observación inicial.

        Retorno
        =======

        observation: object
            La observación inicial del espacio.
        """

        self._validate_agents()
        self._counter = 0
        ob = self._get_observation()

        return ob

    def render(self) -> None:
        """Interfaz gráfica del environment."""

        if self._fig is None or self._axes is None:
            self._fig, self._axes = plt.subplots(ncols=2, 
                                                 figsize=(12.8,4.8))
        
        _pnl = pd.DataFrame(columns=self.agents.keys(),
                            index=self.dates)
        
        for agent in self.agents:
            _pnl[agent] = (self.agents[agent].rewards.sum(axis=1)+1).cumprod()
        
        self._axes[0].clear()
        self._axes[1].clear()

        self._prices.loc[:self.index].plot(ax=self._axes[0])
        _pnl.loc[:self.index].plot(ax=self._axes[1])

        self._axes[0].set_xlim(self.dates.min(), self.dates.max())
        self._axes[0].set_title('Precios de mercado')
        self._axes[0].set_ylabel('Precios')
        self._axes[1].set_xlim(self._pnl.index.min(), self._pnl.index.max())
        self._axes[1].set_title('PnL')
        self._axes[1].set_ylabel('Nivel de riqueza')
        plt.pause(0.0001)
        self._fig.canvas.draw()

    def summary(self) -> pd.DataFrame:
        """
        Genera resumen de estadísticas y figuras.

        Retorno
        =======

        table: pd.DataFrame
            Reporte de la estratégia.
        """
        summary = {}

        for agent in self.agents:
            prices = self._prices 
            returns = self.agents[agent].rewards.sum(axis=1)
            returns.name = agent 
            weights = self.agents[agent].actions 
            weights.name = agent 
            summary[agent] = poptim.utils.summary.stats(returns)
            poptim.utils.summary.figure(prices, returns, weights)
        
        return pd.DataFrame(summary)

        