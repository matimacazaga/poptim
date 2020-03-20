import numpy as np 

eps = np.finfo(float).eps 

def _check_shapes(w: np.ndarray, mu: np.ndarray,
                 Sigma: np.ndarray, w0: np.ndarray):
    
    assert Sigma.shape[0] == Sigma.shape[1]
    assert mu.shape[0] == Sigma.shape[0]
    assert w.shape == w0.shape

def _mu_p(w: np.ndarray, r: np.ndarray) -> float:
    return np.dot(w.T, r)

def _sigma_p(w: np.ndarray, Sigma: np.ndarray) -> float:
    return np.dot(np.dot(w.T, Sigma), w)

def _trans_costs(w: np.ndarray, w0: np.ndarray, coef: float) -> float:
    return np.sum(np.abs(w0-w))*coef 

def risk_aversion(w: np.ndarray, mu: np.ndarray,
                  Sigma: np.ndarray, w0: np.ndarray,
                  alpha: float, beta: float) -> float: 

    _check_shapes(w, mu, Sigma, w0)

    return - (_mu_p(w, mu) - alpha * _sigma_p(w, Sigma) - _trans_costs(w, w0, beta))

def sharpe_ratio(w: np.ndarray, mu: np.ndarray,
                 Sigma: np.ndarray, w0: np.ndarray,
                 beta: float) -> float:

    return - ((_mu_p(w, mu) - _trans_costs(w, w0, beta)) / (_sigma_p(w, Sigma) + eps))