# Pathophysiology Simulator Architecture

## System Overview
The Pathophysiology Simulator is a high-performance, modular platform designed for the simulation of complex human physiological and pathological systems. It utilizes a state-of-the-art symbolic compilation pipeline and adaptive numerical solvers to handle stiff ODE systems with thousands of variables.

## Core Modules

### 1. Model Data Layer (`models/`)
-   **Schema**: Pydantic-based validation of variables, parameters, and equations.
-   **Specification**: AI-compatible JSON format for defining physiological compartments.

### 2. Symbolic Compiler (`compiler/`)
-   **Symbolic Validator**: Uses SymPy for recursive dimensional analysis and NetworkX for detecting algebraic loops in the model dependency graph.
-   **JIT Compiler**: Transforms symbolic SymPy expressions into optimized Python code strings, which are then JIT-compiled using **Numba** for near-native execution speed.

### 3. Numerical Engine (`core/`)
-   **State Vector**: Contiguous NumPy array management with O(1) named indexing.
-   **Adaptive Solver**: SciPy-based `solve_ivp` supporting **BDF** and **LSODA** methods with sparse Jacobian utilization for high-dimensional efficiency.
-   **EventManager**: A sophisticated root-finding loop that detects physiological triggers, applies instantaneous state/parameter jumps, and restarts the solver to maintain numerical integrity.

### 4. Advanced Analytics (`core/`)
-   **SensitivityAnalysis**: Computes normalized finite-difference sensitivities and ranks parameters by influence.
-   **SteadyStateFinder**: Identifies system equilibria using non-linear root-finding.
-   **Bifurcation Module**: Scans parameter spaces to track fixed points and eigenvalue stability.

### 5. Visualization Suite (`visualization/`)
-   **Streamlit Dashboard**: Real-time research interface for parameter tuning and comparative analysis.
-   **Plotly/Plotnine**: Publication-quality static and interactive graphics.
-   **Manim Animator**: Automated mathematical animations of model topology and feedback loops.
