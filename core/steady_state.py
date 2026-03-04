import numpy as np
from scipy.optimize import root
from typing import Optional, Tuple
from core.engine import Engine

class SteadyStateFinder:
    def __init__(self, engine: Engine):
        self.engine = engine

    def find_steady_state(self, y_guess: Optional[np.ndarray] = None, t: float = 0.0) -> Tuple[np.ndarray, bool]:
        """Finds a steady state (equilibrium) where dy/dt = 0."""
        if y_guess is None:
            y_guess = self.engine.state_vector.get_vector()

        params = self.engine.parameters.get_vector()

        def objective(y):
            return self.engine.rhs(t, y, params)

        res = root(objective, y_guess, method='hybr')

        if res.success:
            return res.x, True
        else:
            return res.x, False
