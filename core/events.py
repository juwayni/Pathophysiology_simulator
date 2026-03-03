import numpy as np
import sympy as sp
from typing import Callable, List, Dict, Optional, Tuple, Any
from models.schema import ModelSchema, EventSchema, VariableType

class EventManager:
    def __init__(self, model: ModelSchema, state_vector, parameters):
        self.model = model
        self.state_vector = state_vector
        self.parameters = parameters
        self.events = model.events
        self.compiled_events = []
        self._compile_events()

    def _compile_events(self):
        # Create symbols for all state variables and parameters
        state_vars = [v.name for v in self.model.variables if v.type == VariableType.STATE]
        param_vars = [p.name for p in self.model.parameters]
        all_symbols = {name: sp.Symbol(name) for name in state_vars + param_vars}
        all_symbols['t'] = sp.Symbol('t')

        for event in self.events:
            # Condition: a function that returns 0 when the condition is met (event root finding)
            cond_expr = sp.sympify(event.condition, locals=all_symbols)
            cond_func = sp.lambdify((all_symbols['t'], [all_symbols[v] for v in state_vars], [all_symbols[p] for p in param_vars]), cond_expr, modules='numpy')

            # Action: a function that applies a change to the state or parameters
            action_parts = event.action.split('=')
            target_var = action_parts[0].strip()
            action_expr = sp.sympify(action_parts[1].strip(), locals=all_symbols)
            action_func = sp.lambdify((all_symbols['t'], [all_symbols[v] for v in state_vars], [all_symbols[p] for p in param_vars]), action_expr, modules='numpy')

            # Create the event_root function with a way to access current parameters
            # Since solve_ivp doesn't pass params to events, we'll store them in self.parameters
            # and access them in the event_root.

            def make_event_root(c_func, mgr):
                def event_root(t, y):
                    # Always get the latest parameters from the manager
                    params_vec = mgr.parameters.get_vector()
                    return c_func(t, y, params_vec)
                return event_root

            root_func = make_event_root(cond_func, self)
            root_func.terminal = True
            root_func.direction = 1

            self.compiled_events.append({
                'name': event.name,
                'root': root_func,
                'target': target_var,
                'action_func': action_func
            })

    def get_event_roots(self) -> List[Callable]:
        return [e['root'] for e in self.compiled_events]

    def apply_event_action(self, event_idx: int, t: float, y: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        event = self.compiled_events[event_idx]
        new_val = event['action_func'](t, y, params)

        target = event['target']
        new_y = y.copy()
        new_params = params.copy()

        # Check if it's a state variable or parameter
        if target in self.state_vector.name_to_idx:
            new_y[self.state_vector.name_to_idx[target]] = new_val
        elif target in self.parameters.name_to_idx:
            new_params[self.parameters.name_to_idx[target]] = new_val
        else:
            raise ValueError(f"Unknown target {target} in event {event['name']}")

        return new_y, new_params
