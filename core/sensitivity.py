import numpy as np
import scipy.linalg
from typing import Callable, List, Dict, Optional, Tuple
from core.engine import Engine

class SensitivityAnalysis:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.model = engine.model

    def compute_local_sensitivity(self, t: float, delta: float = 1e-6) -> np.ndarray:
        """Computes local sensitivity of each state variable derivative with respect to each parameter."""
        n_states = self.engine.state_vector.n_states
        n_params = self.engine.parameters.n_params

        y = self.engine.state_vector.get_vector()
        params = self.engine.parameters.get_vector()

        # Original derivative
        dy_base = self.engine.rhs(t, y, params)

        sensitivity_matrix = np.zeros((n_states, n_params))

        for i in range(n_params):
            params_pert = params.copy()
            params_pert[i] += delta
            dy_pert = self.engine.rhs(t, y, params_pert)
            sensitivity_matrix[:, i] = (dy_pert - dy_base) / delta

        return sensitivity_matrix

    def analyze_stability(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Analyzes stability of the system at time t by computing the eigenvalues of the Jacobian."""
        y = self.engine.state_vector.get_vector()
        params = self.engine.parameters.get_vector()

        # Compute Jacobian at current state
        J = self.engine.jac(t, y, params)

        # Eigenvalues
        eigenvalues = scipy.linalg.eigvals(J)

        # If all real parts < 0, stable
        is_stable = np.all(np.real(eigenvalues) < 0)

        return eigenvalues, is_stable
