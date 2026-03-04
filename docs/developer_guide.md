# Developer Guide

## Core Philosophy
The simulator is designed for extensibility. New solvers, mathematical primitives, or visualization types should be added as modular plugins without modifying the core simulation loop.

## Development Setup
1.  **Environment**: Python 3.10+
2.  **Dependencies**:
    ```bash
    pip install numpy scipy sympy numba pandas plotly plotnine streamlit networkx pydantic pytest manim
    ```
3.  **Pathing**: Always run from the root directory with `export PYTHONPATH=$PYTHONPATH:.`.

## Modifying the Compiler
The `JITCompiler` generates code strings. If you need to support new mathematical functions (e.g., custom sigmoids):
1.  Update the `functional_primitives` list in `models/schema.py`.
2.  Update the `_get_expression_dimension` method in `compiler/symbolic_validator.py`.
3.  Ensure the Numba environment can resolve the function name.

## Adding Backend Support
To support JAX or Torch backends:
1.  Create a new class in `core/solver.py` inheriting from the base logic.
2.  Implement the vectorized derivative evaluation using the specific framework's JIT.

## Testing Standards
-   All new features must include an integration test in `tests/`.
-   Scaling benchmarks should be run to ensure no performance regressions in the Jacobian sparsity logic.
