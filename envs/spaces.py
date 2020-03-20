import numpy as np 
import gym 

class PortfolioVector(gym.Space):
    """Estructura de datos para el vector de un portfolio."""

    def __init__(self, num_instruments):
        """
        Construye un objeto 'PortfolioVector'.

        Parámetros
        ==========

        num_instruments: int 
            Cantidad de acciones (cardinalidad del universo),
        """

        self.low = np.zeros(num_instruments, dtype=float) 
        self.high = np.ones(num_instruments, dtype=float) * np.inf 

    def sample(self):
        """Extrae muestra aleatoria de 'PortfolioVector' (pesos)."""
        _vec = np.random.uniform(0, 1.0, self.shape[0])
        return _vec / np.sum(_vec)

    def contains(self, x, tolerance=1e-5):
        """Corrobora si 'x' está en el espacio."""
        shape_predicate = x.shape == self.shape 
        range_predicate = (x >= self.low).all() and (x <= self.high).all()
        budget_constraint = np.abs(x.sum() - 1.0) < tolerance 
        return shape_predicate and range_predicate and budget_constraint

    @property
    def shape(self):
        """Forma del objeto 'PortfolioVector'."""
        return self.low.shape 

    def __repr__(self):
        return f'PortfolioVector {self.shape}'

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)    