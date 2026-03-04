# Mathematical Foundations

## ODE Systems
The simulator handles systems of the form:
$$\frac{dy}{dt} = f(t, y, p)$$
where $f$ is the JIT-compiled RHS.

## Jacobian and Sparsity
The compiler automatically computes the symbolic Jacobian:
$$J = \frac{\partial f}{\partial y}$$
The system detects the sparsity pattern of $J$ using the variable dependency graph, enabling efficient integration of large-scale systems.

## Stability Analysis
Stability is determined by the eigenvalues of the Jacobian at a steady state $y^*$.
-   **Asymptotically Stable**: All $\text{Re}(\lambda) < 0$.
-   **Unstable**: Any $\text{Re}(\lambda) > 0$.

## Recursive Dimensional Analysis
The simulator recursively tracks physical dimensions ($L, T, M, P, V$) through complex symbolic expressions to ensure mathematical consistency.
-   **Homogeneity**: Additions and subtractions must have matching dimensions.
-   **Consistency**: RHS derivative dimensions must match LHS derivative definitions.
