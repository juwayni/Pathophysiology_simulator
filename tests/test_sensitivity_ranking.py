import numpy as np
from core.engine import Engine
from core.sensitivity import SensitivityAnalysis
from models.schema import ModelSchema

def test_sensitivity_ranking():
    model_data = {
        "metadata": {"name": "Sens Model", "version": "1.0"},
        "variables": [
            {"name": "x", "unit": "1", "initial_value": 1.0, "type": "state"}
        ],
        "parameters": [
            {"name": "k1", "unit": "1", "value": 1.0},
            {"name": "k2", "unit": "1", "value": 0.1}
        ],
        "equations": [
            {"target": "x", "expression": "-k1 * x + k2"}
        ]
    }
    model = ModelSchema(**model_data)
    engine = Engine(model)
    sa = SensitivityAnalysis(engine)

    ranking = sa.get_sensitivity_ranking(0.0)
    print(f"Ranking: \n{ranking}")

    assert ranking['parameter'].iloc[0] == 'k1'
    assert ranking['influence'].iloc[0] > ranking['influence'].iloc[1]
    print("Sensitivity ranking test passed!")

if __name__ == "__main__":
    test_sensitivity_ranking()
