# Mathematical Foundations of the Pathophysiology Simulator

## Governing Equations
The system models physiology as a system of Ordinary Differential Equations (ODEs):
$$\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}, \mathbf{p})$$
where:
- $\mathbf{y} \in \mathbb{R}^n$ is the state vector (e.g., pressures, volumes, concentrations).
- $\mathbf{p} \in \mathbb{R}^m$ is the parameter vector (e.g., resistances, compliances, gains).
- $\mathbf{f}$ is the right-hand side (RHS) function, compiled via SymPy and Numba.

## Numerical Methods

### ODE Solvers
For stiff biological systems, the simulator utilizes the **Backward Differentiation Formula (BDF)** method, which provides stability for systems with widely varying time scales (e.g., nerve signaling vs. hormonal regulation).

### Steady-State Identification
Equilibrium points $\mathbf{y}^*$ are found where $\mathbf{f}(t, \mathbf{y}^*, \mathbf{p}) = \mathbf{0}$. We use the **Modified Powell (hybr)** algorithm from `scipy.optimize.root` for robust convergence.

### Stability and Bifurcation
Stability is assessed via the **Jacobian Matrix**:
$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial y_j}$$
The system is locally asymptotically stable at $\mathbf{y}^*$ if all eigenvalues $\lambda_k$ of $\mathbf{J}(\mathbf{y}^*)$ have negative real parts:
$$\text{Re}(\lambda_k) < 0, \quad \forall k$$
Bifurcation analysis tracks these eigenvalues as parameters $\mathbf{p}$ vary, identifying qualitative changes in behavior (e.g., loss of stability).

## Sensitivity Analysis
We compute local parameter sensitivity $\mathbf{S}$:
$$\mathbf{S}_{ij} = \frac{\partial f_i}{\partial p_j} \approx \frac{f_i(p_j + \Delta p_j) - f_i(p_j)}{\Delta p_j}$$
This identifies which physiological parameters have the most significant impact on system dynamics.
