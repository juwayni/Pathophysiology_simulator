# Mathematical Foundations

## ODE Systems
The simulator handles systems of ordinary differential equations (ODEs) of the form:

$dy/dt = f(t, y, p)$

where:
-   $t$ is time.
-   $y$ is the state vector (variables defined in the model).
-   $p$ is the parameter vector.
-   $f$ is the right-hand side (RHS) function, defined by user equations.

## Solvers
The simulator uses SciPy's `solve_ivp` with:
-   **BDF**: A backward differentiation formula method for stiff systems.
-   **LSODA**: An adaptive solver that switches between Adams and BDF methods.

## Symbolic Jacobian
The JIT compiler computes the symbolic Jacobian of the RHS:

$J_{ij} = \frac{\partial f_i}{\partial y_j}$

This improves solver performance and stability for stiff systems.

## Sensitivity Analysis
Local parameter sensitivity is computed using finite differences:

$S_{ij} = \frac{\partial f_i}{\partial p_j} \approx \frac{f_i(t, y, p + \Delta p_j) - f_i(t, y, p)}{\Delta p_j}$

## Stability Analysis
Stability is assessed via the eigenvalues of the Jacobian at a given state. If all real parts are negative, the system is locally stable.
