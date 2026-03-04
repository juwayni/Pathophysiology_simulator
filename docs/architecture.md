# Pathophysiology Simulator Architecture

## System Overview
The Pathophysiology Simulator is a high-performance, modular platform designed for the simulation of complex human physiological and pathological systems.

## Core Modules

### 1. Backend Abstraction Layer (`core/backend.py`)
Provides a formal interface for numerical solvers.
-   **ScipyBackend**: Reliable adaptive BDF solver for stiff systems.
-   **Future Backends**: Seamless integration for JAX, PyTorch, or GPU-accelerated solvers.

### 2. Symbolic Compiler (`compiler/`)
-   **Symbolic Validator**: PhD-level recursive dimensional analysis and algebraic loop detection.
-   **JIT Compiler**: High-performance code generation using SymPy and **Numba**.

### 3. Numerical Engine (`core/engine.py`)
-   Orchestrates simulation with contiguous state vectors and a robust **Event Handling** loop.
-   Supports sparse Jacobian patterns for O(1000) variable systems.

### 4. Advanced Analytics (`core/`)
-   **SensitivityAnalysis**: Normalized impact ranking of physiological parameters.
-   **SteadyStateFinder**: Automated equilibrium identification.
-   **Bifurcation**: 1D parameter continuation and stability mapping.

### 5. Visualization Suite (`visualization/`)
-   **Research Studio**: Streamlit-based interface with clinical presets.
-   **Scientific Plots**: Publication-ready Plotly/Plotnine graphics.
-   **Mathematical Animation**: Dramatic Manim-based visualization of physiological feedback loops.
