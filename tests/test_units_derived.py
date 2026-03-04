import pytest
from compiler.symbolic_validator import SymbolicValidator
from models.schema import ModelSchema

def test_derived_unit_consistency():
    model_data = {
        "metadata": {"name": "Derived Model", "version": "1.0"},
        "variables": [
            {"name": "P", "unit": "mmHg", "initial_value": 100.0, "type": "state"},
            {"name": "Q", "unit": "mL/min", "initial_value": 5000.0, "type": "algebraic"}
        ],
        "parameters": [
            {"name": "R", "unit": "mmHg*min/mL", "value": 0.02}
        ],
        "equations": [
            # P [mmHg] = Q [mL/min] * R [mmHg*min/mL] -> Correct
            {"target": "Q", "expression": "P / R"}
        ]
    }
    model = ModelSchema(**model_data)
    validator = SymbolicValidator(model)
    # Check Q = P/R consistency
    expr = validator._get_expression_dimension(validator.symbols['P'] / validator.symbols['R'])
    assert str(expr) == str(validator.dimensions['Q'])
    print("Derived units test passed!")

if __name__ == "__main__":
    test_derived_unit_consistency()
