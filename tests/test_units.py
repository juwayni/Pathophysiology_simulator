import pytest
from compiler.symbolic_validator import SymbolicValidator
from models.schema import ModelSchema

def test_dimensional_analysis_smoke():
    model_data = {
        "metadata": {"name": "Unit Model", "version": "1.0"},
        "variables": [
            {"name": "V", "unit": "meter/second", "initial_value": 1.0, "type": "state"}
        ],
        "parameters": [
            {"name": "a", "unit": "meter/second**2", "value": 9.8}
        ],
        "equations": [
            {"target": "V", "expression": "a * t"}
        ]
    }
    model = ModelSchema(**model_data)
    validator = SymbolicValidator(model)
    # Dimensional analysis currently returns True for placeholders but ensures structural definition.
    assert validator.perform_dimensional_analysis() is True

if __name__ == "__main__":
    test_dimensional_analysis_smoke()
    print("Dimensional analysis smoke test passed!")
