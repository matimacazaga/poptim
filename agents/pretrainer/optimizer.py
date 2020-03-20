import numpy as np 
import scipy.optimize as sp_opt

def optimizer(J, *args):

    def _optimizer(mu: np.ndarray, Sigma: np.ndarray,
                   w0: np.ndarray, allow_short: bool = True) -> np.ndarray:

        # Numero de acciones
        M = mu.shape[0]

        # Restriccion (igualdad) ==> sum(w) = 1
        con_budget = {'type': 'eq',
                      'fun': lambda w: np.sum(w) - 1.}

        if allow_short:

            bounds = [(None, None) for _ in range(M)]

        else:

            bounds = [(0, None) for _ in range(M)]

        results = sp_opt.minimize(J, w0,
                                 (mu, Sigma, w0, *args),
                                 constraints=(con_budget),
                                 bounds=bounds,
                                 method='SLSQP')

        if not results.success:
            raise BaseException(results.message)

        w = results.x

        return w 

    return _optimizer
