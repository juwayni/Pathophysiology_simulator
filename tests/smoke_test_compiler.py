import numpy as np
import sympy as sp
from numba import njit
from models.schema import ModelSchema, ModelMetadata, VariableSchema, VariableType, ParameterSchema, EquationSchema
from compiler.jit_compiler import JITCompiler
from compiler.symbolic_validator import SymbolicValidator

def smoke_test_compiler():
    # Define a simple SIR model
    model_data = {
        "metadata": {
            "name": "SIR Model",
            "version": "1.0",
            "description": "Simple SIR epidemiological model"
        },
        "variables": [
            {"name": "S", "unit": "people", "initial_value": 999.0, "type": "state"},
            {"name": "I", "unit": "people", "initial_value": 1.0, "type": "state"},
            {"name": "R", "unit": "people", "initial_value": 0.0, "type": "state"}
        ],
        "parameters": [
            {"name": "beta", "unit": "1/day", "value": 0.3},
            {"name": "gamma", "unit": "1/day", "value": 0.1},
            {"name": "N", "unit": "people", "value": 1000.0}
        ],
        "equations": [
            {"target": "S", "expression": "-beta * S * I / N"},
            {"target": "I", "expression": "beta * S * I / N - gamma * I"},
            {"target": "R", "expression": "gamma * I"}
        ]
    }

    model = ModelSchema(**model_data)
    print("Model validated by Pydantic.")

    validator = SymbolicValidator(model)
    validator.validate_equations()
    print("Equations validated by SymbolicValidator.")

    compiler = JITCompiler(model)
    rhs = compiler.compile_rhs()
    print("RHS compiled successfully.")

    # Test RHS
    t = 0.0
    y = np.array([999.0, 1.0, 0.0])
    params = np.array([0.3, 0.1, 1000.0])
    dy = rhs(t, y, params)
    print(f"RHS output at t=0: {dy}")

    # Expected output:
    # dS/dt = -0.3 * 999 * 1 / 1000 = -0.2997
    # dI/dt = 0.2997 - 0.1 * 1 = 0.1997
    # dR/dt = 0.1 * 1 = 0.1
    expected = np.array([-0.2997, 0.1997, 0.1])
    np.testing.assert_allclose(dy, expected, atol=1e-6)
    print("Smoke test passed!")

if __name__ == "__main__":
    smoke_test_compiler()
