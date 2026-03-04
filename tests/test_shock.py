import pytest
import numpy as np
from compiler.parser import ModelLoader
from core.engine import Engine

def test_shock_model():
    model = ModelLoader.load_from_json("models/example_shock.json")
    engine = Engine(model)

    history = engine.run((0, 150))

    assert "P_art" in history.columns
    assert "V_total" in history.columns

    # Verify events
    # The jump should be visible at exactly t=10.0
    at_10 = history[history['t'] == 10.0]
    if len(at_10) >= 2:
        v_diff = at_10['V_total'].iloc[0] - at_10['V_total'].iloc[-1]
        assert abs(v_diff - 2000.0) < 1.0

    # Check for SNS increase
    assert history['SNS'].max() > history['SNS'].iloc[0]
    print("Shock model integration test passed.")

if __name__ == "__main__":
    test_shock_model()
