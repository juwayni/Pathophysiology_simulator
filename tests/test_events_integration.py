import numpy as np
import pandas as pd
from models.schema import ModelSchema
from core.engine import Engine

def test_events_integration():
    # Define SIR model with an event at t=50
    model_data = {
        "metadata": {"name": "SIR with Event", "version": "1.0"},
        "variables": [
            {"name": "S", "unit": "people", "initial_value": 990.0, "type": "state"},
            {"name": "I", "unit": "people", "initial_value": 10.0, "type": "state"},
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
        ],
        "events": [
            {
                "name": "Social Distancing",
                "condition": "t - 50.0",
                "action": "beta = 0.05",
                "description": "Beta drops at t=50"
            }
        ]
    }

    model = ModelSchema(**model_data)
    engine = Engine(model)

    t_span = (0, 100)
    history = engine.run(t_span)

    print("Simulation with event complete.")

    # Check that beta change affected S trajectory
    # Before t=50, S should drop faster
    # After t=50, S should drop much slower

    s_at_50 = history[history['t'] >= 50.0].iloc[0]['S']
    s_at_end = history.iloc[-1]['S']

    # S at 50 vs S at 100
    print(f"S(50): {s_at_50}, S(100): {s_at_end}")

    assert s_at_end < s_at_50 # Should still be decreasing
    # But slope should be very small

    # Check if params were updated
    assert engine.parameters.get_value("beta") == 0.05
    print("Event integration test passed!")

if __name__ == "__main__":
    test_events_integration()
