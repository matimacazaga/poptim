import numpy as np
import pandas as pd 

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar 'pandas.DataFrame' de valores 
    nulos.

    Parámetros
    ==========

    df: pandas.DataFrame
        Tabla a ser limpiada.
    
    Retorno
    =======

    clean_df: pandas.DataFrame
        Tabla sin valores nulos.
    """

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def align(target, source):
    """
    Alinear índices, columnas y nombres.

    Parámetros
    ==========

    target: numpy.ndarray
        Objecto a ser alineado
    source: pandas.Series | pandas.DataFrame
        Objeto plantilla utilizado para alinear.
    
    Retorno
    =======

    obj: pandas.Series | pandas.DataFrame
        Objecto pandas alineado.
    """

    if target.shape != source.shape:
        raise ValueError('Shapes diferentes, imposible alinear')

    if isinstance(source, pd.Series):
        obj = pd.Series(target, index=source.index, name=source.name)
    elif isinstance(source, pd.DataFrame):
        obj = pd.DataFrame(target, index=source.index, columns=source.columns)
    else:
        raise ValueError('Tipo de "source" no reconocido')
    
    return obj