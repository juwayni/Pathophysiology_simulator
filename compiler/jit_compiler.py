import sympy as sp
import numpy as np
from numba import njit
from typing import List, Dict, Callable
from models.schema import ModelSchema, VariableType

class JITCompiler:
    def __init__(self, model: ModelSchema):
        self.model = model
        self.state_vars = [v.name for v in model.variables if v.type == VariableType.STATE]
        self.param_vars = [p.name for p in model.parameters]
        self.n_states = len(self.state_vars)
        self.n_params = len(self.param_vars)

    def _generate_func_code(self, func_name: str, expressions: List[sp.Expr], is_matrix: bool = False) -> str:
        # Generate Numba-compatible Python code string from expressions
        lines = [f"@njit\ndef {func_name}(t, y, params):"]

        # Unpack y
        for i, name in enumerate(self.state_vars):
            lines.append(f"    {name} = y[{i}]")

        # Unpack params
        for i, name in enumerate(self.param_vars):
            lines.append(f"    {name} = params[{i}]")

        # Add math functions to namespace
        # (Numba handles most np functions directly)

        if is_matrix:
            # Jacobian is a matrix
            rows, cols = expressions.shape
            lines.append(f"    out = np.zeros(({rows}, {cols}), dtype=np.float64)")
            for r in range(rows):
                for c in range(cols):
                    expr_str = str(expressions[r, c]).replace('t', 't') # Ensure t is t
                    # Handle power operator and potential other SymPy vs Python differences
                    # Note: SymPy's str(expr) uses ** which Python/Numba understands.
                    if expr_str != '0':
                        lines.append(f"    out[{r}, {c}] = {expr_str}")
            lines.append("    return out")
        else:
            # RHS is a vector
            lines.append(f"    out = np.zeros({len(expressions)}, dtype=np.float64)")
            for i, expr in enumerate(expressions):
                expr_str = str(expr)
                if expr_str != '0':
                    lines.append(f"    out[{i}] = {expr_str}")
            lines.append("    return out")

        return "\n".join(lines)

    def compile_rhs(self) -> Callable:
        """Compiles the Right-Hand Side (RHS) using Numba JIT."""
        state_symbols = {name: sp.Symbol(name) for name in self.state_vars}
        param_symbols = {name: sp.Symbol(name) for name in self.param_vars}
        all_symbols = {**state_symbols, **param_symbols, 't': sp.Symbol('t')}

        rhs_expressions = []
        for var_name in self.state_vars:
            eq = next((e for e in self.model.equations if e.target == var_name), None)
            rhs_expressions.append(sp.sympify(eq.expression, locals=all_symbols) if eq else sp.Float(0.0))

        code = self._generate_func_code("rhs_numba", rhs_expressions)

        # Define namespace for exec
        namespace = {'np': np, 'njit': njit}
        # Add common math functions that might be in expressions
        # Actually SymPy's str(expr) might use sin, cos etc which are in np namespace if we use np.sin etc
        # But for direct code, it might use 'sin(x)'.
        # Let's ensure sin/cos etc are mapped to np.sin/np.cos
        code = code.replace('sin(', 'np.sin(').replace('cos(', 'np.cos(').replace('exp(', 'np.exp(').replace('log(', 'np.log(').replace('sqrt(', 'np.sqrt(').replace('abs(', 'np.abs(')

        exec(code, namespace)
        return namespace['rhs_numba']

    def compile_jacobian(self) -> Callable:
        """Compiles the Jacobian using Numba JIT."""
        state_symbols = [sp.Symbol(name) for name in self.state_vars]
        param_symbols = [sp.Symbol(name) for name in self.param_vars]
        all_symbols = {**{s.name: s for s in state_symbols}, **{p.name: p for p in param_symbols}, 't': sp.Symbol('t')}

        rhs_expressions = []
        for var_name in self.state_vars:
            eq = next((e for e in self.model.equations if e.target == var_name), None)
            rhs_expressions.append(sp.sympify(eq.expression, locals=all_symbols) if eq else sp.Float(0.0))

        jacobian_matrix = sp.Matrix(rhs_expressions).jacobian(state_symbols)

        code = self._generate_func_code("jac_numba", jacobian_matrix, is_matrix=True)
        namespace = {'np': np, 'njit': njit}
        code = code.replace('sin(', 'np.sin(').replace('cos(', 'np.cos(').replace('exp(', 'np.exp(').replace('log(', 'np.log(').replace('sqrt(', 'np.sqrt(').replace('abs(', 'np.abs(')

        exec(code, namespace)
        return namespace['jac_numba']
