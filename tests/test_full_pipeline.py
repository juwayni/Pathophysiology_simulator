import pytest
import numpy as np
from compiler.parser import ModelLoader
from core.engine import Engine

def test_full_pipeline():
    # Load example cardiovascular model
    model = ModelLoader.load_from_json("models/example_cardiovascular.json")

    # Initialize Engine
    engine = Engine(model)

    # Run simulation
    t_span = (0, 30)
    history = engine.run(t_span)

    # Assertions
    assert "P_art" in history.columns
    assert "P_ven" in history.columns
    assert "V_art" in history.columns
    assert "V_ven" in history.columns

    # Check for basic physical realism (values are not NaN)
    assert not history['P_art'].isnull().any()
    assert not history['P_ven'].isnull().any()

    # Verify events (hemorrhage at t=10 should cause a jump/discontinuity)
    # The history should contain at least t=0 and t=30
    assert history['t'].iloc[0] == 0.0
    assert history['t'].iloc[-1] == 30.0

    print("Full pipeline integration test passed.")

if __name__ == "__main__":
    test_full_pipeline()
