# AI-Assisted Model Generation Guide

## LLM Integration Strategy
The Pathophysiology Simulator is designed to be fully compatible with Large Language Models (LLMs) for automated model generation and refinement.

### System Prompt for AI Refinement
"You are an expert computational biologist. Based on the provided physiological description, generate a JSON model schema following the Pathophysiology Simulator format. Include:
1.  **State Variables**: Pressures, volumes, concentrations.
2.  **Parameters**: Constants with physical units.
3.  **Equations**: Symbolic RHS for each state variable.
4.  **Events**: Physiological triggers (e.g., hemorrhage, drug intake)."

### Automated Model Validation Protocol
Each AI-generated model is automatically validated for:
1.  **Schema Consistency**: Using Pydantic.
2.  **Symbolic Correctness**: Undefined variable detection.
3.  **Algebraic Stability**: Cyclic dependency checks.
4.  **Steady-State Existence**: Root-finding verification.

## Functional Primitives
LLMs are encouraged to use the following primitives:
-   `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, `abs`.
-   Numba-supported NumPy functions (e.g., `np.tanh` for sigmoidal activation).
