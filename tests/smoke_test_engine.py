import numpy as np
import pandas as pd
from models.schema import ModelSchema
from core.engine import Engine

def smoke_test_engine():
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
    engine = Engine(model)

    t_span = (0, 160)
    t_eval = np.linspace(0, 160, 161)

    history = engine.run(t_span, t_eval=t_eval)

    print("Simulation complete.")
    print(f"Final state: {history.iloc[-1]}")

    # Check if S decreases and R increases
    assert history['S'].iloc[-1] < history['S'].iloc[0]
    assert history['R'].iloc[-1] > history['R'].iloc[0]
    print("Smoke test passed!")

if __name__ == "__main__":
    smoke_test_engine()
