import streamlit as st
import numpy as np
import json
import pandas as pd
from models.schema import ModelSchema
from core.engine import Engine
from core.sensitivity import SensitivityAnalysis
from visualization.plotly_visualizer import PlotlyVisualizer
from visualization.network_graph import NetworkGraph

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

    model_json = st.sidebar.text_area("Model JSON", json.dumps(default_model, indent=2), height=200)

    try:
        model_dict = json.loads(model_json)
        model = ModelSchema(**model_dict)
        engine = Engine(model)

        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Sensitivity & Stability", "Structure", "Raw Data"])

        with tab1:
            st.sidebar.subheader("Simulation Parameters")
            t_max = st.sidebar.slider("Simulation Time (days)", 10, 500, 160)

            for param in model.parameters:
                new_val = st.sidebar.number_input(f"{param.name} ({param.unit})", value=float(param.value), key=f"p_{param.name}")
                engine.set_parameter(param.name, new_val)

            if st.sidebar.button("Run Simulation"):
                t_span = (0, t_max)
                history = engine.run(t_span)

                st.header(f"Simulation Results: {model.metadata.name}")
                vars_to_plot = st.multiselect("Plot Variables", [v.name for v in model.variables], default=[v.name for v in model.variables])
                fig = PlotlyVisualizer.plot_time_series(history, vars_to_plot)
                st.plotly_chart(fig, use_container_width=True)

                st.session_state['history'] = history
                st.session_state['engine'] = engine

        with tab2:
            if 'engine' in st.session_state:
                st.header("Sensitivity & Stability Analysis")
                sa = SensitivityAnalysis(st.session_state['engine'])
                t_sa = st.number_input("Analysis Time", value=0.0)

                # Stability
                eig, stable = sa.analyze_stability(t_sa)
                st.subheader(f"System Stability at t={t_sa}")
                st.write(f"Is Locally Stable: {'Yes' if stable else 'No'}")
                st.write(f"Eigenvalues: {eig}")

                # Sensitivity
                sens = sa.compute_local_sensitivity(t_sa)
                st.subheader("Local Parameter Sensitivity")
                sens_df = pd.DataFrame(sens,
                                       index=[v.name for v in model.variables if v.type == 'state'],
                                       columns=[p.name for p in model.parameters])
                st.table(sens_df)
            else:
                st.info("Run simulation first.")

        with tab3:
            st.header("Dependency Graph")
            ng = NetworkGraph(engine.validator.get_dependency_graph())
            fig_graph = ng.plot_interactive_graph()
            st.plotly_chart(fig_graph, use_container_width=True)

        with tab4:
            if 'history' in st.session_state:
                st.dataframe(st.session_state['history'])

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
