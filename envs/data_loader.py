import numpy as np 
import pandas as pd 
import os 
import typing
import pandas_datareader as pdr 
import urllib.request 

class Finance:
    """ Wrapper para datos del mercado"""

    def __init__(self, mkt):

        self.mkt = mkt
        self._col = 'cierre' if self.mkt == 'ARG' else 'Adj Close'

    def _get_arg(self, ticker: str, **kwargs) -> typing.Optional[pd.DataFrame]:        
        """
        Método para obtener datos de rava
        bursátil (mercado Argentino).

        Parámetros
        ==========
        
        ticker: str
            Nombre de la acción.
        **kwargs: dict
            * start: str, pandas.Timestamp
                Fecha inicial
            * end: str, pandas.Timestamp
                Fecha final
        
        Retorno
        =======

        df: pandas.DataFrame
            Datos para la acción 'ticker'.        
        """
        try:
            df = pd.read_csv(f'https://www.rava.com/empresas/precioshistoricos.php?e={ticker}&csv=1')
            df.loc[:,'fecha'] = pd.to_datetime(df.loc[:,'fecha'],
                                               format='%Y-%m-%d')
            df.set_index('fecha', drop=True, inplace=True)
            df = df.loc[kwargs.get('start'):kwargs.get('end'), :]
            return df
        except:
            print(f'Fallo al descargar datos para {ticker}')
            return None 
    
    def _get_usa(self, ticker: str, **kwargs):
        """
        Método para obtener datos de yahoo
        finance (mercado Norteamericano).

        Parámetros
        ==========
        
        ticker: str
            Nombre de la acción.
        **kwargs: dict
            * start: str, pandas.Timestamp
                Fecha inicial.
            * end: str, pandas.Timestamp
                Fecha final.
        
        Retorno
        =======

        df: pandas.DataFrame
            Datos para la acción 'ticker'.        
        """
        try:
            df = pdr.DataReader(ticker, 'yahoo',
                                start=kwargs.get('start'),
                                end=kwargs.get('end'))
            return df
        except:
            print(f'Fallo al descargar datos para {ticker}')
            return None
    
    def _get(self, ticker, **kwargs) -> typing.Optional[pd.DataFrame]:
        """
        Método para obtener datos del
        mercado.

        Parámetros
        ==========

        ticker: str
            Nombre de la acción
        **kwargs: dict
            * start: str, pandas.Timestamp
                Fecha inicial.
            * end: str, pandas.Timestamp
                Fecha final.
        
        Retorno
        =======

        df: pandas.DataFrame
            Datos para la acción 'ticker'.
        """

        if self.mkt == 'ARG':
            return self._get_arg(ticker, **kwargs) 
        elif self.mkt == 'USA':
            return self._get_usa(ticker, **kwargs)
        else:
            print(f'Mercado {self.mkt} no reconocido')
            return None

    def _csv(self, root: str, tickers: typing.Union[str, typing.List[str]]):
        """
        Método para cargar precios desde
        archivos CSV .

        Parámetros
        ==========

        root: str
            Ruta al archivo CSV.
        tickers: str, list[str]
            Ticker o lista de tickers.

        Retorno
        =======

        df: pandas.DataFrame
            Data del archivo CSV.
        """
        
        df = pd.read_csv(root, index_col='Date',
                         parse_dates=True).sort_index(ascending=True)
        
        union = [ticker for ticker in tickers if ticker in df.columns]
        return df[union]

    def Returns(self, tickers: typing.List[str], csv: str = None, **kwargs):

        if isinstance(csv, str):
            return self._csv(csv, tickers).loc[kwargs.get('start'):kwargs.get('end'),:]
        else:
            return self.Prices(tickers, **kwargs).pct_change()[1:]

    def Prices(self, tickers: typing.List[str], csv: str = None, **kwargs):
        """
        Método para obtener los precios de
        los 'tickers'.
        """
        if isinstance(csv, str):
            return self._csv(csv, tickers).loc[kwargs.get('start'):kwargs.get('end'), :]
        else:
            data = {}
            for ticker in tickers:
                tmp_df = self._get(ticker, **kwargs)
                if tmp_df is not None:
                    data[ticker] = tmp_df[self._col]
            df = pd.DataFrame(data)
            return df.sort_index(ascending=True)

if __name__ == '__main__':

    fin = Finance('USA')
    data = fin.Returns(['MSFT', 'AAPL'], start='2015', end='2018')
    print(data.head())





