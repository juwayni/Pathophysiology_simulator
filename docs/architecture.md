# Pathophysiology Simulator Architecture

## Overview
The Pathophysiology Simulator is a modular, high-performance, and extensible platform for simulating human physiology and pathology using dynamical systems modeling.

## Core Components

1.  **Model Schema (`models/schema.py`)**: Defines the simulation model structure (variables, parameters, equations, units) using Pydantic.
2.  **Symbolic Compiler (`compiler/`)**:
    -   `parser.py`: Loads and validates model JSON.
    -   `symbolic_validator.py`: Uses SymPy for equation validation, dimensional analysis, and Jacobian sparsity pattern detection.
    -   `jit_compiler.py`: Compiles symbolic equations into high-performance RHS and Jacobian functions using **Numba JIT**.
3.  **Core Engine (`core/`)**:
    -   `state_vector.py`: Efficient memory management of simulation state.
    -   `parameters.py`: Parameter storage and update logic.
    -   `solver.py`: Wrapper for SciPy `solve_ivp` supporting BDF/LSODA.
    -   `engine.py`: Orchestrates the simulation process, including a robust **Event Handling** loop.
    -   `events.py`: Implements a root-finding based event system for instantaneous state jumps and parameter changes.
    -   `sensitivity.py`: Finite difference sensitivity analysis and stability analysis.
4.  **Visualization (`visualization/`)**:
    -   `plotly_visualizer.py`: Interactive scientific plots.
    -   `network_graph.py`: Dependency visualization using NetworkX.
    -   `dashboard_streamlit.py`: Interactive research dashboard.
    -   `plotnine_publication.py`: Publication-ready plots using ggplot2-like syntax.
    -   `manim_animator.py`: Mathematical animations using Manim.

## Design Principles
-   Strict separation between model data, mathematical compilation, and solver backend.
-   Variable-centric dynamical architecture.
-   Support for large-scale stiff ODE systems through JIT and sparsity.
-   AI-compatible JSON model schemas.
