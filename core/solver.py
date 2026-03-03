import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from typing import Callable, List, Optional, Tuple, Dict
from models.schema import ModelSchema

class Solver:
    def __init__(self, rhs: Callable, jac: Optional[Callable] = None, jac_sparsity: Optional[csr_matrix] = None, method: str = 'BDF', atol: float = 1e-6, rtol: float = 1e-6):
        self.rhs = rhs
        self.jac = jac
        self.jac_sparsity = jac_sparsity
        self.method = method
        self.atol = atol
        self.rtol = rtol

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float], params: np.ndarray, t_eval: Optional[np.ndarray] = None, events: Optional[List[Callable]] = None):
        """Wrapper for scipy solve_ivp with sparsity support."""
        def rhs_wrapped(t, y):
            return self.rhs(t, y, params)

        def jac_wrapped(t, y):
            if self.jac:
                return self.jac(t, y, params)
            return None

        return solve_ivp(
            rhs_wrapped,
            t_span,
            y0,
            method=self.method,
            jac=jac_wrapped if self.jac else None,
            jac_sparsity=self.jac_sparsity,
            t_eval=t_eval,
            events=events,
            atol=self.atol,
            rtol=self.rtol
        )
