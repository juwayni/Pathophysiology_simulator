import numpy as np
from core.engine import Engine
from core.sensitivity import SensitivityAnalysis
from models.schema import ModelSchema

def test_bifurcation_smoke():
    model_data = {
        "metadata": {"name": "Equilibrium Model", "version": "1.0"},
        "variables": [
            {"name": "x", "unit": "1", "initial_value": 0.5, "type": "state"}
        ],
        "parameters": [
            {"name": "k", "unit": "1", "value": 1.0}
        ],
        "equations": [
            {"target": "x", "expression": "-k * x"}
        ]
    }
    model = ModelSchema(**model_data)
    engine = Engine(model)
    sa = SensitivityAnalysis(engine)

    results = sa.scan_bifurcation("k", (0.5, 2.0), n_steps=5)

    assert len(results) == 5
    for res in results:
        assert abs(res['steady_state'][0]) < 1e-4
        assert np.real(res['eigenvalues'][0]) < 0
        assert bool(res['is_stable']) is True

    print("Bifurcation and Steady-state smoke test passed!")

if __name__ == "__main__":
    test_bifurcation_smoke()
