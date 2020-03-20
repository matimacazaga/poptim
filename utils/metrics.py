import numpy as np 
import pandas as pd 
import scipy.stats as sp_stats

eps = np.finfo(float).eps 

def cum_returns(returns):
    """
    Calcula los retornos cumulativos a 
    partir de los retornos simples.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======

    cumulative_returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos cumulativos.
    """

    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)
    out = np.subtract(out, 1)
    return pd.Series(out) 

def pnl(returns):
    """
    Calcula el PnL a partir de los retornos
    simples.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos d ela estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======

    pnl: np.ndarray | pd.Series | pd.DataFrame
        PnL.
    """

    if returns.ndim > 2:
        raise ValueError('Tensor de retornos no puede ser utilizado')

    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)
    return out 

def sharpe_ratio(returns):
    """
    Calcula el Ratio de Sharpe a partir
    de los retornos simples.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como 
        porcentaje, no cumulativos.
    
    Retorno
    =======
    
    sharpe_ratio : float | np.ndarray | pd.Series
        Ratio de Sharpe.
    """

    if returns.ndim > 2:
        raise ValueError('Tensor de retornos no puede ser utilizado')
    
    return np.mean(returns, axis=0) / (np.std(returns, axis=0))

def probabilistic_sharpe_ratio(returns, benchmark_sr):
    """
    Calcula el ratio de Sharpe probabilistico.
    (Pág. 203, Advances in Financial ML, López de Prado)

    Parámetros
    ==========

    returns: np.ndarray | pd.Series
        Serie de retornos como porcentaje, 
        no cumulativo.
    benchmark_sr: float
        Valor de ratio de Sharpe utilizado 
        como benchmark. Si es nulo, se compara
        contra "ninguna habilidad de invertir"
    
    Retorno
    =======
    
    float
        Ratio probabilístico de Sharpe.

    """
    sr = sharpe_ratio(returns)
    numerator = (sr - benchmark_sr) * np.sqrt(len(returns) - 1.)  
    denominator = np.sqrt(1. - skewness(returns) * sr + 0.25*(sp_stats.kurtosis(returns.values, fisher=False)-1.)* sr**2)
    x = numerator/denominator
    return sp_stats.norm.cdf(x)

def deflated_sharpe_ratio(returns, srs):
    """
    Calcula el ratio de Sharpe desviado.
    (Pág. 204, Advances in Financial ML, López de Prado)

    Parámeters
    ==========

    srs: list | np.array | pd.Series 
        Conjunto de estimaciones del ratio 
        de Sharpe.

    Retorno
    =======

    float
        Ratio de Sharpe probabilístico, 
        utilizando una estimación de 
        SR* como benchmark.
    """
    a = np.sqrt(np.var(srs, ddof=1))
    b = (1. - np.euler_gamma) * sp_stats.norm.ppf(1.-(1./len(srs)))
    c = np.euler_gamma * sp_stats.norm.ppf(1. - (1./len(srs))*np.exp(-1) )
    sr = a * ( b + c )
    return probabilistic_sharpe_ratio(returns, sr) 

def compute_dd_tuw(series, dollars=False):
    """
    Calcula la serie de "drawdowns" y el "time
    under water" asociado. 
    (Pág. 201, Advances in Financial ML, López de Prado)

    Parámetros
    ==========

    series: pd.Series
        Serie de retornos o performance en dólares.
    dollars: bool
        Si dollars == True, la serie es de performance
        en dólares. Si dollars == False, es de retornos.

    Retorno
    =======

    dd: pd.DataFrame
        Drawdowns.
    tuw: pd.Series
        Time under water.
    """
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index
    df1 = df1[df1['hwm'] > df1['min']]
    if dollars:
        dd = df1['hwm']-df1['min']
    else:
        dd = 1-df1['min']/df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1,'Y')).values # Años 
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw

def get_hhi(returns):
    """
    Concentración de retornos.
    (Pág. 200, Advances in Financial ML, Lópes de Prado)

    Puede calcularse para retornos negativos, positivos, o 
    por frecuencia (ej.: concentración mensual).

    Parámetros
    ==========
    
    returns: pd.Series | np.array
        Retornos de la estrategia como
        porcentaje, no cumulativo.
    
    Retorno
    =======

    hhi: float
        Concentración de retornos.
    """
    if returns.shape[0] <= 2:
        return np.nan 
    wght = returns/returns.sum()
    hhi = (wght**2).sum()
    hhi = (hhi - returns.shape[0]**(-1))/(1. - returns.shape[0]**(-1))
    return hhi 


def hit_ratio(returns):
    """
    Calcula el Hit Ratio a partir de 
    los retornos simples, representado por
    el número de trades positivos sobre el 
    total de trades realizados.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como 
        porcentaje, no cumulativos.
    
    Retorno
    =======

    hit_ratio: float | np.ndarray | pd.Series
        Hit ratio.
    """

    if returns.ndim > 2:
        raise ValueError('Tensor de retornos no puede ser utilizado')
    
    return np.sum(returns > 0, axis=0) / len(returns)

def awal(returns):
    """
    Calcula el Ratio entre la ganancia 
    promedio y la pérdida promedio.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como
        porcentaje, no cumulativos.
    
    Retorno
    =======

    awal: float | np.ndarray | pd.Series
        Ratio entre ganancia promedio y 
        pérdida promedio.
    """  

    if returns.ndim > 2:
        raise ValueError('Tensor de retornos no puede ser utilizado')
    
    aw = returns[returns>0].mean(axis=0)
    al = returns[returns<0].mean(axis=0)
    return np.abs((aw+eps)/(al+eps))

def appt(returns):
    """
    Calcula la ganancia promedio por trade.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======
    
    appt: float | np.ndarray | pd.Series
        Ganancia promedio por trade.
    """

    if returns.ndim > 2:
        raise ValueError('Tensor de retornos no puede ser utilizado')
    
    pw = np.sum(returns > 0, axis=0) / len(returns)
    pl = np.sum(returns < 0, axis=0) / len(returns)
    aw = returns[returns>0].mean(axis=0)
    al = returns[returns<0].mean(axis=0)
    return (pw * aw) - (pl * al)

def drawdown(returns):
    """
    Calcula el Drawdown a partir de
    los retornos simples.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======

    drawdown: pandas.Series
        Drawdown de la estrategia.
    """

    _cum_returns = cum_returns(returns)
    expanding_max = _cum_returns.expanding(1).max()
    drawdown = expanding_max - _cum_returns
    return drawdown

def max_drawdown(returns):
    """
    Calcula el máximo Drawdown a partir de
    los retornos simples.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======

    max_drawdown: pandas.Series
        Máximo Drawdown de la estategia.
    """

    _drawdown = drawdown(returns)
    return _drawdown.expanding(1).max()

def average_drawdown_time(returns):
    """
    Calcula el tiempo promedio de Drawdown a
    partir de los retornos simples.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no comulativos.
    
    Retorno
    =======

    average_drawdown_time: datetime.timedelta
        Tiempo promedio de Drawdown de la 
        estrategia. 
    """

    _drawdown = drawdown(returns)
    return _drawdown[_drawdown == 0].index.to_series().diff().mean()

def mean_returns(returns):
    """
    Calcula el retorno promedio a partir
    de los retornos simples.

    Párametros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativo.

    Retorno
    =======

    mean_returns: float
        Retorno promedio de la estrategia.
    """

    return returns.mean(axis=0)

def std_returns(returns):
    """
    Calcula la desviación estándar de los 
    retornos a partir de los retornos
    simples.

    Parámetros
    ==========
    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativo.

    Retorno
    =======
    std_returns: float
        Desviación estándar de los retornos
        de la estrategia.
    """
    return returns.std(axis=0)

def skewness(returns):
    """
    Calcula la asimetría de los retornos a
    partir de los retornos simples.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativos.

    Retorno
    =======

    skew_returns: float
        Asimetría de los retornos de la 
        estrategia.
    """

    return returns.skew(axis=0)

def kurtosis(returns):
    """
    Calcula la kurtosis de los retornos a
    partir de los retornos simples.

    Parámetros
    ==========
    
    returns: pandas.Series
        Retornos de la estrategia como 
        porcentaje, no cumulativo.

    Retorno
    =======

    kurt_returns: float
        Kurtosis de la estrategia.
    """

    return returns.kurt(axis=0)

def tail_ratio(returns):
    """
    Calcula el ratio entre el percentil-95 y
    el percentil-5 (valor absoluto) de los 
    retornos simples (tail ratio).

    Por ejemplo, un tail ratio de 0.25
    significa que las pérdidas son cuatro
    veces tan malas como las ganancias.
    
    Parámetros
    ==========

    return: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativo.

    Retorno
    =======

    tail_ratio: float
        'Tail ratio' de los retornos
        de la estrategia.
    """

    return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))

def value_at_risk(returns, cutoff=0.05):
    """
    Calcula el VaR de los retornos.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como
        porcentaje, no cumulativo.
    cutoff: float, optional
        Decimal representando el porcentaje
        de corte para el percentil inferior
        de los retornos.
    
    Retorno
    =======

    VaR: float
        The VaR value.
    """

    return np.percentile(returns, 100.*cutoff)

def conditional_value_at_risk(returns, cutoff=0.05):
    """
    Calcula el CVaR de los retornos.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como 
        porcentaje, no cumulativo.
    cutoff: float, optional
        Decimal representando el porcentaje
        de corte para el percentil inferior 
        de los retornos.
    
    Retorno
    =======

    CVaR: float
        CVaR.
    """

    cutoff_index = int((len(returns)-1)*cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])