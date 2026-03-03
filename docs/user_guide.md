# User Guide

## Getting Started

1.  **Define Your Model**: Create a JSON file following the schema in `models/schema.py`. See `models/example_cardiovascular.json` for inspiration.
2.  **Initialize the Engine**: Load the model and create an instance of the simulation engine.
    ```python
    from compiler.parser import ModelLoader
    from core.engine import Engine

    model = ModelLoader.load_from_json("path/to/your_model.json")
    engine = Engine(model)
    ```
3.  **Run Simulation**: Run the simulation over a given time span.
    ```python
    t_span = (0, 100)
    t_eval = np.linspace(0, 100, 101)
    history = engine.run(t_span, t_eval=t_eval)
    ```
4.  **Visualize Results**: Use the interactive Plotly visualizer or the Streamlit dashboard.
    ```bash
    streamlit run visualization/dashboard_streamlit.py
    ```

## Customizing Parameters
You can modify parameters on-the-fly:
```python
engine.set_parameter("beta", 0.05)
engine.run((0, 50))
```

## Adding Events
Events can be defined in the model JSON:
```json
{
  "name": "Treatment",
  "condition": "t - 50.0",
  "action": "beta = 0.05"
}
```
This event will trigger at `t=50s`, setting the parameter `beta` to `0.05`.
