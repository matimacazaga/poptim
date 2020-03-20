import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import poptim.utils.econometric as pue

def time_series(series, title='', xlabel='', ylabel='', path=None):
    """
    Plot de series de tiempo univariadas y multivariadas.

    Parámetros
    ==========

    series: np.ndarray | pd.Series | pd.DataFrame
        Serie univariada/multivariada para plotear.
    title: str, optional
        Título de la figura.
    xlabel: str, optional
        Etiqueta para el eje x.
    ylabel: str, optional
        Etiqueta para el eje y.
    path: str, optional
        Carpeta para guardar la figura.
    """

    fig, ax = plt.subplot()
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        series.plot(ax=ax)
    elif isinstance(series, np.ndarray):
        if series.ndim == 1:
            ax.plot(series)
        elif series.ndim == 2:
            for c in range(series.shape[1]):
                ax.plot(series[:,c])
        else:
            raise ValueError('Tensor no puede ser manejado')
    
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
        
    ax.xaxis.set_tick_params(rotation=45)
    if path is not None:
        fig.savefig(path)
    fig.show()

def pnl(returns, path=None):
    """
    Plot de PnL para retornos simples.

    Parámetros
    ==========

    returns: np.ndarray | pd.Series | pd.DataFrame
        Retornos de la estrategia como porcentaje, no cumulativo.
    path: str, optional
        Carpeta para guardar la figura.
    """

    _pnl = pue.pnl(returns)
    if hasattr(returns, 'name'):
        title = f'{returns.name or "Estrategia"}: Profit & Loss'
    else:
        title = 'Profit & Loss'
    
    xlabel = 'Tiempo'
    ylabel = 'PnL'
    time_series(_pnl, title, xlabel, ylabel, path)

def trades(prices, weights, path=None):
    """
    Plot de precios de las acciones y los correspondientse
    pesos del portfolio.

    Parámetros
    ==========

    prices: pandas.Series
        Precio de los activos.
    weights: pandas.Series
        Pesos del portfolio.
    path: str, optional
        Carpeta para guardar la figura.
    """

    fig, axes = plt.subplots(nrows=2, sharex=True,
                             gridspec_kw={'height_ratios':[3,1],
                                          'wspace': 0.01})
    axes[0].plot(prices.index, prices.values, color='b')
    axes[1].bar(weights.index, weights.values, color='g')
    axes[0].set(title=f'{prices.name or "Estrategia"}: Precios y Pesos del portoflio',
                ylabel='Precio $p_{t}$')
    axes[1].set(xlabel='Tiempo', ylabel='Pesos $w_{t}$')
    axes[1].xaxis.set_tick_params(rotation=45)
    fig.subplots_adjust(hspace=.0)
    if path is not None:
        fig.savefig(path)
    fig.show()

def table_image(array, path=None):
    """
    Plot de data 2d como imagen.

    Parámetros
    ==========

    array: numpy.ndarray | pandas.DataFrame
        Data 2d a graficar.
    path: str, optional
        Carpeta para guardar la figura.
    """

    if array.ndim != 2:
        raise ValueError('Array debe ser 2d')
    fig, ax = plt.subplots()
    ax.imshow(array, cmap=plt.get_cmap('Greys'))
    ax.axis('off')
    if path is not None:
        fig.savefig(path)
    fig.show()

def drawdown(returns, path=None):
    """
    Plot del Drawdown junto con PnL.

    Parámetros
    ==========

    returns: pandas.Series
        Retornos de la estrategia como porcentaje, no cumulativo.
    path: str, optional
        Carpeta para guardar la figura.
    """

    pnl = pue.pnl(returns)
    neg_drawdown = - pue.drawdown(returns)
    neg_max_drawdown = - pue.max_drawdown(returns)
    fig, ax = plt.subplots(figsize=(12,8))
    pnl.plot(label='Profit & Loss', ax=ax)
    neg_drawdown.plot(label='Drawdown', ax=ax)
    neg_max_drawdown.plot(label='Max Drawdown', ax=ax)
    ax.set(title=f'{returns.name or "Estrategia"}: Profit & Loss con Drawdown',
           ylabel='PnL', xlabel='Tiempo')
    ax.legend()
    if path is not None:
        fig.savefig(path)