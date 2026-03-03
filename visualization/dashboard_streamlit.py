import streamlit as st
import numpy as np
import json
from models.schema import ModelSchema
from core.engine import Engine
from visualization.plotly_visualizer import PlotlyVisualizer

def main():
    st.set_page_config(page_title="Pathophysiology Simulator Dashboard", layout="wide")
    st.title("Human Physiology and Pathology Simulator")

    # Model Loading
    st.sidebar.header("Model Configuration")

    # Default SIR model
    default_model = {
        "metadata": {"name": "SIR Model", "version": "1.0"},
        "variables": [
            {"name": "S", "unit": "people", "initial_value": 999.0, "type": "state"},
            {"name": "I", "unit": "people", "initial_value": 1.0, "type": "state"},
            {"name": "R", "unit": "people", "initial_value": 0.0, "type": "state"}
        ],
        "parameters": [
            {"name": "beta", "unit": "1/day", "value": 0.3},
            {"name": "gamma", "unit": "1/day", "value": 0.1},
            {"name": "N", "unit": "people", "value": 1000.0}
        ],
        "equations": [
            {"target": "S", "expression": "-beta * S * I / N"},
            {"target": "I", "expression": "beta * S * I / N - gamma * I"},
            {"target": "R", "expression": "gamma * I"}
        ]
    }

    model_json = st.sidebar.text_area("Model JSON", json.dumps(default_model, indent=2), height=300)

    try:
        model_dict = json.loads(model_json)
        model = ModelSchema(**model_dict)
        engine = Engine(model)

        # Simulation parameters
        st.sidebar.subheader("Simulation Parameters")
        t_max = st.sidebar.slider("Simulation Time (days)", 10, 500, 160)

        # Parameter tuning
        st.sidebar.subheader("Tune Parameters")
        for param in model.parameters:
            new_val = st.sidebar.number_input(f"{param.name} ({param.unit})", value=float(param.value))
            engine.set_parameter(param.name, new_val)

        if st.sidebar.button("Run Simulation"):
            t_span = (0, t_max)
            t_eval = np.linspace(0, t_max, t_max + 1)
            history = engine.run(t_span, t_eval=t_eval)

            st.header(f"Simulation Results: {model.metadata.name}")

            # Time Series
            vars_to_plot = st.multiselect("Select Variables to Plot", [v.name for v in model.variables], default=[v.name for v in model.variables])
            fig = PlotlyVisualizer.plot_time_series(history, vars_to_plot)
            st.plotly_chart(fig, use_container_width=True)

            # Phase Space
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X-axis", [v.name for v in model.variables], index=0)
                y_var = st.selectbox("Y-axis", [v.name for v in model.variables], index=1)
                fig_phase = PlotlyVisualizer.plot_phase_space(history, x_var, y_var)
                st.plotly_chart(fig_phase, use_container_width=True)
            with col2:
                st.subheader("Final State")
                st.table(history.iloc[-1])

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
