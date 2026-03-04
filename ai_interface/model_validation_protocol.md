# AI Model Validation Protocol

All AI-generated models for the Pathophysiology Simulator must adhere to this protocol before deployment.

## 1. Schema Validation
-   Must conform to the Pydantic `ModelSchema`.
-   All variables must have specified physical units (SI or PHYSIO).
-   Initial conditions must be within realistic physiological ranges.

## 2. Symbolic Integrity
-   No undefined symbols in equations.
-   **Dimensional Homogeneity**: Recursive check that LHS and RHS dimensions match.
-   **Algebraic Solvability**: No cycles in algebraic variable dependencies.

## 3. Numerical Stability
-   **Baseline Check**: Baseline Jacobian must not contain NaN/Inf.
-   **Local Stability**: At least one stable equilibrium point must be identifiable within a ±20% perturbation of parameters.

## 4. Performance
-   Systems > 500 variables must provide a dependency graph for sparse Jacobian optimization.
-   All equations must use supported functional primitives (Numba-compatible).
