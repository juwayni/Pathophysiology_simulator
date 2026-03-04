import numpy as np
import scipy.linalg
from typing import Callable, List, Dict, Optional, Tuple
from core.engine import Engine
from core.steady_state import SteadyStateFinder

class SensitivityAnalysis:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.ss_finder = SteadyStateFinder(engine)

    def compute_local_sensitivity(self, t: float, delta: float = 1e-6) -> np.ndarray:
        n_states = self.engine.state_vector.n_states
        n_params = self.engine.parameters.n_params
        y = self.engine.state_vector.get_vector()
        params = self.engine.parameters.get_vector()

        dy_base = self.engine.rhs(t, y, params)
        sensitivity_matrix = np.zeros((n_states, n_params))

        for i in range(n_params):
            p_pert = params.copy()
            p_pert[i] += delta
            dy_pert = self.engine.rhs(t, y, p_pert)
            sensitivity_matrix[:, i] = (dy_pert - dy_base) / delta
        return sensitivity_matrix

    def analyze_stability(self, t: float, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        if y is None:
            y = self.engine.state_vector.get_vector()
        params = self.engine.parameters.get_vector()
        J = self.engine.jac(t, y, params)
        eigenvalues = scipy.linalg.eigvals(J)
        # For a stable equilibrium, real parts must be negative.
        is_stable = np.all(np.real(eigenvalues) < 0)
        return eigenvalues, is_stable

    def scan_bifurcation(self, param_name: str, p_range: Tuple[float, float], n_steps: int = 50) -> List[Dict]:
        p_vals = np.linspace(p_range[0], p_range[1], n_steps)
        orig_val = self.engine.parameters.get_value(param_name)
        results = []
        y_guess = self.engine.state_vector.get_vector()

        for p in p_vals:
            self.engine.set_parameter(param_name, p)
            y_ss, success = self.ss_finder.find_steady_state(y_guess)
            if success:
                y_guess = y_ss
                eig, stable = self.analyze_stability(0.0, y_ss)
                results.append({
                    'param_value': p,
                    'steady_state': y_ss,
                    'eigenvalues': eig,
                    'is_stable': stable
                })
        self.engine.set_parameter(param_name, orig_val)
        return results
