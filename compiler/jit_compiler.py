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

    def compile_rhs(self) -> Callable:
        """Compiles the Right-Hand Side (RHS) using SymPy's lambdify with Numba."""
        state_symbols = [sp.Symbol(name) for name in self.state_vars]
        param_symbols = [sp.Symbol(name) for name in self.param_vars]
        t_symbol = sp.Symbol('t')

        all_symbols_dict = {**{s.name: s for s in state_symbols}, **{p.name: p for p in param_symbols}, 't': t_symbol}

        rhs_expressions = []
        for var_name in self.state_vars:
            eq = next((e for e in self.model.equations if e.target == var_name), None)
            rhs_expressions.append(sp.sympify(eq.expression, locals=all_symbols_dict) if eq else sp.Float(0.0))

        # Use lambdify with numba backend
        # Note: we want a function (t, y, params)
        # solve_ivp passes (t, y) and we wrap it to pass params.
        # lambdify( (t, y_vec, p_vec), exprs, 'numba' )

        # We need to be careful with vector arguments in Numba lambdify.
        # Often it's better to lambdify with individual arguments or use a specific printer.

        # Actually, let's use the improved code generation approach but make it more robust
        # by using SymPy's Python printer instead of simple str().

        from sympy.printing.pycode import PythonCodePrinter
        printer = PythonCodePrinter()

        lines = ["import numpy as np", "from numba import njit", "@njit", "def rhs_numba(t, y, params):"]
        for i, name in enumerate(self.state_vars):
            lines.append(f"    {name} = y[{i}]")
        for i, name in enumerate(self.param_vars):
            lines.append(f"    {name} = params[{i}]")

        lines.append(f"    out = np.zeros({self.n_states}, dtype=np.float64)")
        for i, expr in enumerate(rhs_expressions):
            expr_str = printer.doprint(expr)
            lines.append(f"    out[{i}] = {expr_str}")
        lines.append("    return out")

        code = "\n".join(lines)
        namespace = {}
        exec(code, namespace)
        return namespace['rhs_numba']

    def compile_jacobian(self) -> Callable:
        """Compiles the Jacobian using SymPy's symbolic differentiation and Numba."""
        state_symbols = [sp.Symbol(name) for name in self.state_vars]
        param_symbols = [sp.Symbol(name) for name in self.param_vars]
        t_symbol = sp.Symbol('t')
        all_symbols_dict = {**{s.name: s for s in state_symbols}, **{p.name: p for p in param_symbols}, 't': t_symbol}

        rhs_expressions = []
        for var_name in self.state_vars:
            eq = next((e for e in self.model.equations if e.target == var_name), None)
            rhs_expressions.append(sp.sympify(eq.expression, locals=all_symbols_dict) if eq else sp.Float(0.0))

        jacobian_matrix = sp.Matrix(rhs_expressions).jacobian(state_symbols)

        from sympy.printing.pycode import PythonCodePrinter
        printer = PythonCodePrinter()

        lines = ["import numpy as np", "from numba import njit", "@njit", "def jac_numba(t, y, params):"]
        for i, name in enumerate(self.state_vars):
            lines.append(f"    {name} = y[{i}]")
        for i, name in enumerate(self.param_vars):
            lines.append(f"    {name} = params[{i}]")

        rows, cols = jacobian_matrix.shape
        lines.append(f"    out = np.zeros(({rows}, {cols}), dtype=np.float64)")
        for r in range(rows):
            for c in range(cols):
                val = jacobian_matrix[r, c]
                if val != 0:
                    expr_str = printer.doprint(val)
                    lines.append(f"    out[{r}, {c}] = {expr_str}")
        lines.append("    return out")

        code = "\n".join(lines)
        namespace = {}
        exec(code, namespace)
        return namespace['jac_numba']
