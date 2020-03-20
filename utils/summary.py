import pandas as pd 
import matplotlib.pyplot as plt 
import poptim.utils 


def stats(returns):
    """
    Genera reporte de estadísticas para la
    estrategia.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos realizados de la estrategia.

    Retorno
    =======

    table: pd.Series
        Reporte de la estrategia.
    """

    report = {
        'Mean Returns': poptim.utils.econometric.mean_returns(returns),
        'Cumulative Returns': poptim.utils.econometric.cum_returns(returns).iloc[-1],
        'Volatility': poptim.utils.econometric.std_returns(returns),
        'Sharpe Ratio': poptim.utils.econometric.sharpe_ratio(returns),
        'Max Drawdown': poptim.utils.econometric.max_drawdown(returns).iloc[-1],
        'Average Drawdown Time': poptim.utils.econometric. average_drawdown_time(returns).days,
        'Skewness': poptim.utils.econometric.skewness(returns),
        'Kurtosis': poptim.utils.econometric.kurtosis(returns),
        'Tail ratio': poptim.utils.econometric.tail_ratio(returns),
        'VaR': poptim.utils.econometric.value_at_risk(returns),
        'CVaR': poptim.utils.econometric.conditional_value_at_risk(returns),
        'Hit Ratio': poptim.utils.econometric.hit_ratio(returns),
        'Average Win to Average Loss': poptim.utils.econometric.awal(returns),
        'Average Profitability per Trade': poptim.utils.econometric.appt(returns) 
    }

    table = pd.Series(report, name=(returns.name or 'Strategy'), 
                      dtype=object)
    
    return table 

def figure(prices, returns, weights, path=None):
    """
    Genera figuras estadísticas para la estrategia.

    Parámetros
    ==========

    prices: pandas.DataFrame
        Precios de los activos del universo.
    returns: pandas.Series
        Retornos realizados de la estrategia.
    weights: pandas.DataFrame
        Pesos del portfolio correspondientes
        a la estrategia.
    path: str, optional
        Ruta para guardar la figura.
    """

    poptim.utils.plotting.drawdown(returns, path)
    for ticker in prices:
        poptim.utils.plotting.trades(prices[ticker], weights[ticker], path)