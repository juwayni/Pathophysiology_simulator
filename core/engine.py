import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from models.schema import ModelSchema, VariableType
from core.state_vector import StateVector
from core.parameters import Parameters
from core.solver import Solver
from core.events import EventManager
from core.backend import ScipyBackend
from compiler.jit_compiler import JITCompiler
from compiler.symbolic_validator import SymbolicValidator

class Engine:
    def __init__(self, model: ModelSchema, backend=None):
        self.model = model
        self.state_vector = StateVector(model)
        self.parameters = Parameters(model)

        self.validator = SymbolicValidator(model)
        self.validator.validate_equations()
        self.validator.perform_dimensional_analysis()

        self.compiler = JITCompiler(model)
        self.rhs = self.compiler.compile_rhs()
        self.jac = self.compiler.compile_jacobian()
        self.jac_sparsity = self.validator.get_sparsity_pattern()

        self.validator.validate_baseline_stability(self.jac, self.state_vector.get_vector(), self.parameters.get_vector())

        # Use provided backend or default to Scipy
        self.backend = backend or ScipyBackend()
        self.solver = Solver(self.rhs, self.jac, jac_sparsity=self.jac_sparsity, backend=self.backend)

        self.event_manager = EventManager(model, self.state_vector, self.parameters)
        self.history = None

    def run(self, t_span: Tuple[float, float]) -> pd.DataFrame:
        """Main simulation run with backend abstraction and event handling."""
        y_current = self.state_vector.get_vector()
        params_current = self.parameters.get_vector()
        t_start, t_end = t_span

        all_times = []
        all_states = []

        current_t = t_start
        all_times.append(current_t)
        all_states.append(y_current.copy())

        max_iter = 1000
        iters = 0

        while current_t < t_end and iters < max_iter:
            iters += 1
            sol = self.solver.solve(
                y_current, (current_t, t_end), params_current,
                events=self.event_manager.get_event_roots()
            )

            if len(sol.t) > 1:
                start_idx = 1 if abs(sol.t[0] - all_times[-1]) < 1e-12 else 0
                all_times.extend(sol.t[start_idx:])
                all_states.extend(sol.y[:, start_idx:].T)

            current_t = sol.t[-1]
            y_current = sol.y[:, -1]

            if sol.t_events and any(len(te) > 0 for te in sol.t_events):
                earliest_idx = -1
                earliest_t = float('inf')
                for i, te in enumerate(sol.t_events):
                    if len(te) > 0 and te[0] < earliest_t:
                        earliest_t, earliest_idx = te[0], i

                if earliest_idx != -1:
                    y_current, params_current = self.event_manager.apply_event_action(
                        earliest_idx, current_t, y_current, params_current
                    )
                    self.state_vector.update_from_vector(y_current)
                    self.parameters.update_from_vector(params_current)
                    all_times.append(current_t)
                    all_states.append(y_current.copy())
                    current_t += 1e-9
            else:
                break

        df = pd.DataFrame(all_states, columns=[v.name for v in self.state_vector.state_vars])
        df.insert(0, 't', all_times)
        self.history = df
        return df

    def get_state(self, name: str) -> float:
        return self.state_vector.get_value(name)

    def set_parameter(self, name: str, value: float):
        self.parameters.set_value(name, value)
