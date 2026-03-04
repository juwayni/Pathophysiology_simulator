import pytest
from compiler.symbolic_validator import SymbolicValidator
from models.schema import ModelSchema, VariableType

def test_algebraic_loop():
    model_data = {
        "metadata": {"name": "Loop Model", "version": "1.0"},
        "variables": [
            {"name": "X", "unit": "1", "initial_value": 0.0, "type": "algebraic"},
            {"name": "Y", "unit": "1", "initial_value": 0.0, "type": "algebraic"}
        ],
        "parameters": [],
        "equations": [
            {"target": "X", "expression": "Y + 1"},
            {"target": "Y", "expression": "X + 1"}
        ]
    }
    model = ModelSchema(**model_data)
    validator = SymbolicValidator(model)
    with pytest.raises(ValueError, match="Algebraic loop detected"):
        validator.validate_equations()

def test_unit_consistency_smoke():
    model_data = {
        "metadata": {"name": "Unit Model", "version": "1.0"},
        "variables": [
            {"name": "V", "unit": "m/s", "initial_value": 1.0, "type": "state"}
        ],
        "parameters": [
            {"name": "a", "unit": "m/s^2", "value": 9.8}
        ],
        "equations": [
            {"target": "V", "expression": "a * t"}
        ]
    }
    model = ModelSchema(**model_data)
    validator = SymbolicValidator(model)
    assert validator.perform_dimensional_analysis() is True

if __name__ == "__main__":
    test_algebraic_loop()
    test_unit_consistency_smoke()
    print("Validator tests passed!")
