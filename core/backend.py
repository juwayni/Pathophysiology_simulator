import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Dict, Any
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

class Backend(ABC):
    """Abstract base class for simulation backends (e.g. Scipy, JAX, Torch)."""
    @abstractmethod
    def solve(self,
              rhs: Callable,
              y0: np.ndarray,
              t_span: Tuple[float, float],
              params: np.ndarray,
              jac: Optional[Callable] = None,
              jac_sparsity: Optional[csr_matrix] = None,
              events: Optional[List[Callable]] = None,
              atol: float = 1e-6,
              rtol: float = 1e-6) -> Any:
        pass

class ScipyBackend(Backend):
    """SciPy implementation of the simulation backend."""
    def solve(self,
              rhs: Callable,
              y0: np.ndarray,
              t_span: Tuple[float, float],
              params: np.ndarray,
              jac: Optional[Callable] = None,
              jac_sparsity: Optional[csr_matrix] = None,
              events: Optional[List[Callable]] = None,
              atol: float = 1e-6,
              rtol: float = 1e-6):

        def rhs_wrapped(t, y):
            return rhs(t, y, params)

        def jac_wrapped(t, y):
            if jac:
                return jac(t, y, params)
            return None

        return solve_ivp(
            rhs_wrapped,
            t_span,
            y0,
            method='BDF',
            jac=jac_wrapped if jac else None,
            jac_sparsity=jac_sparsity,
            events=events,
            atol=atol,
            rtol=rtol,
            dense_output=True
        )
