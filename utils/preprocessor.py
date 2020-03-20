import numpy as np 
import pandas as pd 

def rolling1d(series, window):
    """
    Rolling window para serie 1D.

    Parámetros
    ==========
    series: list | numpy.ndarray | pandas.Series
        Data secuencial 1D.
    window: int 
        Tamaño de la ventana.

    Retorno
    =======
    matrix: numpy.ndarray
        Matriz con la serie procesada.
    """

    series = np.array(series)
    if len(series.shape) != 1:
        raise ValueError("El array no es 1D")
    
    shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)

def rolling2d(array, window):
    """
    Rolling window para un array de dos dimensiones 

    Parámetros
    ==========
    array: list | numpy.ndarray | pandas.DataFrame
        Data secuencial en dos dimensiones.
    window: int
        Tamañod de la ventana.

    Retorno
    =======
    matrix: numpy.ndarray
        Matrix de la serie procesada. 
    """

    if isinstance(array, list):
        array = np.array(array)
    if len(array.shape) != 2:
        raise ValueError('El array no es 2D')
    
    out = np.empty((array.shape[0] - window + 1, window, array.shape[1]))
    if isinstance(array, np.ndarray):
        for i, col in enumerate(array.T):
            out[:,:,i] = rolling1d(col, window)
    elif isinstance(array, pd.DataFrame):
        for i, label in enumerate(array):
            out[:,:,i] = rolling1d(array[label], window)
    return out
