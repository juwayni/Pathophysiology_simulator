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

        J = self.engine.jac(t, y, params)
        eigenvalues = scipy.linalg.eigvals(J)
        is_stable = np.all(np.real(eigenvalues) < 0)
        return eigenvalues, is_stable

    def scan_bifurcation(self, param_name: str, p_range: Tuple[float, float], n_steps: int = 50) -> List[Dict]:
        """Performs a basic bifurcation scan by tracking steady-state or eigenvalues over a parameter range."""
        p_vals = np.linspace(p_range[0], p_range[1], n_steps)
        original_val = self.engine.parameters.get_value(param_name)

        results = []
        for p in p_vals:
            self.engine.set_parameter(param_name, p)
            # Re-compile JIT if parameters are baked in (our JIT takes params as array, so no need)
            # Check stability at a reference point (e.g. t=0, y=initial)
            eig, stable = self.analyze_stability(0.0)
            results.append({
                'param_value': p,
                'eigenvalues': eig,
                'is_stable': stable
            })

        # Restore parameter
        self.engine.set_parameter(param_name, original_val)
        return results
