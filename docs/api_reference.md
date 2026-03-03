# API Reference

## Model Schema (`models/schema.py`)
-   `ModelSchema`: Pydantic model for overall simulation state and equations.
-   `VariableSchema`: Pydantic model for state variables.
-   `ParameterSchema`: Pydantic model for constant parameters.
-   `EquationSchema`: Pydantic model for derivative equations.

## Engine (`core/engine.py`)
-   `Engine(model: ModelSchema)`: Orchestrates the simulation process.
    -   `run(t_span, t_eval=None)`: Runs the simulation and returns a Pandas DataFrame.
    -   `set_parameter(name, value)`: Modifies a parameter value.
    -   `get_history()`: Returns the simulation result DataFrame.

## Solver (`core/solver.py`)
-   `Solver(rhs, jac=None, method='BDF')`: Wrapper for SciPy `solve_ivp`.
    -   `solve(y0, t_span, params, t_eval=None, events=None)`: Solves the ODE system.

## JIT Compiler (`compiler/jit_compiler.py`)
-   `JITCompiler(model: ModelSchema)`: Compiles symbolic equations into RHS and Jacobian.
    -   `compile_rhs()`: Returns a callable for the RHS function.
    -   `compile_jacobian()`: Returns a callable for the Jacobian matrix.

## Symbolic Validator (`compiler/symbolic_validator.py`)
-   `SymbolicValidator(model: ModelSchema)`: Validates equations for errors and loops.
    -   `validate_equations()`: Checks for undefined variables and algebraic loops.
    -   `get_dependency_graph()`: Returns a NetworkX DiGraph of variable dependencies.

## Sensitivity Analysis (`core/sensitivity.py`)
-   `SensitivityAnalysis(engine)`: Analyzes local parameter sensitivity and stability.
    -   `compute_local_sensitivity(t, delta=1e-6)`: Computes sensitivity matrix.
    -   `analyze_stability(t)`: Computes eigenvalues of the Jacobian.
