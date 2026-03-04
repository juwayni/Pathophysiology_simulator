import pytest
from compiler.symbolic_validator import SymbolicValidator
from models.schema import ModelSchema

def test_dimensional_inconsistency():
    model_data = {
        "metadata": {"name": "Bad Unit Model", "version": "1.0"},
        "variables": [
            {"name": "x", "unit": "meter", "initial_value": 0.0, "type": "state"}
        ],
        "parameters": [
            {"name": "k", "unit": "second", "value": 1.0}
        ],
        "equations": [
            # d(x)/dt [m/s] = x [m] + k [s] -> Inconsistent addition
            {"target": "x", "expression": "x + k"}
        ]
    }
    model = ModelSchema(**model_data)
    validator = SymbolicValidator(model)
    # Our simple recursive dimension tracker returns None for inconsistent additions
    expr = validator._get_expression_dimension(validator.symbols['x'] + validator.symbols['k'])
    assert expr is None

if __name__ == "__main__":
    test_dimensional_inconsistency()
    print("Advanced unit consistency test passed!")
