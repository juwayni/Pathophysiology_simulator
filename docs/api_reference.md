# API Reference

## Core Engine
### `Engine(model: ModelSchema)`
The primary orchestrator for physiological simulations.
-   `run(t_span: Tuple[float, float]) -> pd.DataFrame`: Executes the simulation with full event handling.
-   `set_parameter(name: str, value: float)`: Updates a model parameter on-the-fly.

## Symbolic Compiler
### `SymbolicValidator(model: ModelSchema)`
Validates model integrity.
-   `validate_equations()`: Checks for undefined symbols and algebraic loops.
-   `perform_dimensional_analysis()`: Recursively verifies dimensional consistency.
-   `get_sparsity_pattern()`: Generates CSR matrix for the sparse Jacobian.

### `JITCompiler(model: ModelSchema)`
Compiles equations to machine code.
-   `compile_rhs()`: Generates Numba-jitted derivative function.
-   `compile_jacobian()`: Generates Numba-jitted Jacobian matrix.

## Analytics
### `SensitivityAnalysis(engine: Engine)`
Tools for system interrogation.
-   `get_sensitivity_ranking(t: float) -> pd.DataFrame`: Ranks parameters by normalized influence.
-   `analyze_stability(t: float) -> Tuple[np.ndarray, bool]`: Computes eigenvalues of the Jacobian.
-   `scan_bifurcation(param, range, steps) -> List[Dict]`: Performs 1D continuation and stability tracking.

## Model Data
### `ModelSchema`
Pydantic model defining variables, parameters, equations, and events.
