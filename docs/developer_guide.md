# Developer Guide

## Development Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/pathophysiology-simulator.git
    cd pathophysiology-simulator
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requirements: numpy, scipy, sympy, numba, pandas, plotly, plotnine, streamlit, networkx, pydantic, pytest)*
3.  **Run Tests**:
    ```bash
    pytest tests/
    ```

## Extending the Core
-   **New Solvers**: Add a wrapper in `core/solver.py` for additional backends (e.g., JAX, torchdiffeq).
-   **New Model Primitives**: Modify `models/schema.py` and `compiler/jit_compiler.py` to support more complex mathematical functions.
-   **Advanced Visualization**: Use `visualization/plotly_visualizer.py` as a base for custom scientific plots.

## Best Practices
-   Follow PEP 8 for Python code.
-   Use type hints for all function signatures.
-   Write unit tests for any new core logic.
-   Keep the model schema and engine logic separate.
-   Avoid global state at all costs.
