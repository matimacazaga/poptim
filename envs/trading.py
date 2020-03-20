from poptim.envs.data_loader import Finance
from poptim.envs.base import BaseEnv
import pandas as pd 

class TradingEnv(BaseEnv):

    def _get_prices(self, universe, **kwargs) -> pd.DataFrame:
        return Finance(self.mkt).Prices(universe, **kwargs)