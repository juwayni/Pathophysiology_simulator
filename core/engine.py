import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from models.schema import ModelSchema, VariableType
from core.state_vector import StateVector
from core.parameters import Parameters
from core.solver import Solver
from core.events import EventManager
from compiler.jit_compiler import JITCompiler
from compiler.symbolic_validator import SymbolicValidator

class Engine:
    def __init__(self, model: ModelSchema):
        self.model = model
        self.state_vector = StateVector(model)
        self.parameters = Parameters(model)

        # Validation and Compilation
        self.validator = SymbolicValidator(model)
        self.validator.validate_equations()
        self.validator.perform_dimensional_analysis()
        self.jac_sparsity = self.validator.get_sparsity_pattern()

        self.compiler = JITCompiler(model)
        self.rhs = self.compiler.compile_rhs()
        self.jac = self.compiler.compile_jacobian()

        # Solver with sparsity support
        self.solver = Solver(self.rhs, self.jac, jac_sparsity=self.jac_sparsity)

        self.event_manager = EventManager(model, self.state_vector, self.parameters)
        self.history = None

    def run(self, t_span: Tuple[float, float], t_eval: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Main simulation run with event handling."""
        y_current = self.state_vector.get_vector()
        params_current = self.parameters.get_vector()
        t_start, t_end = t_span

        all_times = []
        all_states = []

        current_t = t_start
        max_iter = 100
        iters = 0

        while current_t < t_end and iters < max_iter:
            iters += 1
            event_roots = self.event_manager.get_event_roots()

            sol = self.solver.solve(
                y_current,
                (current_t, t_end),
                params_current,
                events=event_roots
            )

            all_times.extend(sol.t)
            all_states.extend(sol.y.T)

            current_t = sol.t[-1]
            y_current = sol.y[:, -1]

            if sol.t_events and any(len(te) > 0 for te in sol.t_events):
                earliest_event_idx = -1
                earliest_time = float('inf')
                for i, te in enumerate(sol.t_events):
                    if len(te) > 0 and te[0] < earliest_time:
                        earliest_time = te[0]
                        earliest_event_idx = i

                if earliest_event_idx != -1:
                    y_current, params_current = self.event_manager.apply_event_action(
                        earliest_event_idx, current_t, y_current, params_current
                    )
                    self.state_vector.update_from_vector(y_current)
                    self.parameters.update_from_vector(params_current)
                    current_t += 1e-9
            else:
                break

        col_names = [v.name for v in self.state_vector.state_vars]
        df = pd.DataFrame(all_states, columns=col_names)
        df.insert(0, 't', all_times)
        df = df.drop_duplicates(subset=['t'], keep='last')
        self.history = df
        return df

    def get_history(self) -> Optional[pd.DataFrame]:
        return self.history

    def get_state(self, name: str) -> float:
        return self.state_vector.get_value(name)

    def set_parameter(self, name: str, value: float):
        self.parameters.set_value(name, value)
