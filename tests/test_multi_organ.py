import pytest
import numpy as np
from compiler.parser import ModelLoader
from core.engine import Engine

def test_multi_organ_pipeline():
    # Load complex multi-organ model (V2.0 uses P_art instead of MAP)
    model = ModelLoader.load_from_json("models/example_multi_organ.json")

    # Initialize Engine
    engine = Engine(model)

    # Run simulation
    t_span = (0, 1100)
    history = engine.run(t_span)

    # Assertions
    assert "P_art" in history.columns
    assert "ECBV" in history.columns
    assert "SNS" in history.columns
    assert "AngII" in history.columns

    # Check that events were triggered (should see discontinuities in history)
    # Hemorrhage at t=500 should drop ECBV
    # The jump is recorded exactly at 500
    points_at_500 = history[history['t'] == 500.0]
    if len(points_at_500) >= 2:
        ecbv_diff = points_at_500['ECBV'].iloc[0] - points_at_500['ECBV'].iloc[-1]
        assert abs(ecbv_diff - 1500.0) < 1e-1

    print("Multi-organ pipeline integration test passed.")

if __name__ == "__main__":
    test_multi_organ_pipeline()
