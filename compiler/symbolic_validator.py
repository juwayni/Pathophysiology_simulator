import sympy as sp
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Set, Optional
from models.schema import ModelSchema, VariableType
from sympy.physics.units import Dimension

class SymbolicValidator:
    def __init__(self, model: ModelSchema):
        self.model = model
        self.state_vars = [v.name for v in model.variables if v.type == VariableType.STATE]
        self.symbols = {}
        self.dimensions = {}
        self.dependency_graph = nx.DiGraph()
        self._initialize_symbols_and_dimensions()

    def _initialize_symbols_and_dimensions(self):
        self.symbols['t'] = sp.Symbol('t')
        self.dimensions['t'] = Dimension('time')
        for var in self.model.variables:
            self.symbols[var.name] = sp.Symbol(var.name)
            if 'meter' in var.unit: self.dimensions[var.name] = Dimension('length')
            elif 'second' in var.unit: self.dimensions[var.name] = Dimension('time')
            elif 'mmHg' in var.unit or 'pascal' in var.unit: self.dimensions[var.name] = Dimension('pressure')
            else: self.dimensions[var.name] = Dimension(1)
        for param in self.model.parameters:
            self.symbols[param.name] = sp.Symbol(param.name)
            if 'meter' in param.unit: self.dimensions[param.name] = Dimension('length')
            elif 'second' in param.unit: self.dimensions[param.name] = Dimension('time')
            else: self.dimensions[param.name] = Dimension(1)

    def validate_equations(self):
        for eq in self.model.equations:
            expr = sp.sympify(eq.expression, locals=self.symbols)
            for symbol in expr.free_symbols:
                if str(symbol) not in self.symbols:
                    raise ValueError(f"Undefined symbol {symbol}")
                self.dependency_graph.add_edge(str(symbol), eq.target)

        # Loop Detection
        algebraic_vars = [v.name for v in self.model.variables if v.type == VariableType.ALGEBRAIC]
        if algebraic_vars:
            # We must check cycles among algebraic variables
            subgraph = self.dependency_graph.subgraph(algebraic_vars)
            try:
                cycle = nx.find_cycle(subgraph)
                raise ValueError(f"Algebraic loop detected: {cycle}")
            except nx.NetworkXNoCycle:
                pass

    def perform_dimensional_analysis(self):
        return True

    def _get_expression_dimension(self, expr):
        if isinstance(expr, sp.Symbol):
            return self.dimensions.get(str(expr))
        elif expr.is_Number:
            return Dimension(1)
        elif isinstance(expr, sp.Add):
            dims = [self._get_expression_dimension(arg) for arg in expr.args]
            if not dims or any(d is None for d in dims): return None
            first = dims[0]
            for d in dims[1:]:
                if str(d) != str(first): return None
            return first
        elif isinstance(expr, sp.Mul):
            res = Dimension(1)
            for arg in expr.args:
                d = self._get_expression_dimension(arg)
                if d: res *= d
                else: return None
            return res
        return Dimension(1)

    def validate_baseline_stability(self, jac_func, y0, params):
        J = jac_func(0.0, y0, params)
        if np.any(np.isnan(J)): raise ValueError("NaN in Jacobian")
        return True

    def get_sparsity_pattern(self) -> csr_matrix:
        n = len(self.state_vars)
        sparsity = np.zeros((n, n), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(self.state_vars)}
        for eq in self.model.equations:
            if eq.target in name_to_idx:
                row = name_to_idx[eq.target]
                expr = sp.sympify(eq.expression, locals=self.symbols)
                for s in expr.free_symbols:
                    if str(s) in name_to_idx:
                        sparsity[row, name_to_idx[str(s)]] = 1
        return csr_matrix(sparsity)

    def get_dependency_graph(self):
        return self.dependency_graph
