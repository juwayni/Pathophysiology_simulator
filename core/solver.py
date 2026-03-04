import numpy as np
from typing import Callable, List, Optional, Tuple, Dict
from scipy.sparse import csr_matrix
from core.backend import Backend, ScipyBackend

class Solver:
    def __init__(self, rhs: Callable, jac: Optional[Callable] = None,
                 jac_sparsity: Optional[csr_matrix] = None,
                 backend: Optional[Backend] = None,
                 atol: float = 1e-6, rtol: float = 1e-6):
        self.rhs = rhs
        self.jac = jac
        self.jac_sparsity = jac_sparsity
        self.backend = backend or ScipyBackend()
        self.atol = atol
        self.rtol = rtol

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float], params: np.ndarray,
              events: Optional[List[Callable]] = None):
        """Delegates solving to the backend."""
        return self.backend.solve(
            self.rhs, y0, t_span, params,
            jac=self.jac,
            jac_sparsity=self.jac_sparsity,
            events=events,
            atol=self.atol,
            rtol=self.rtol
        )
