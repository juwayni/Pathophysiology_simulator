import streamlit as st
import numpy as np
import json
import pandas as pd
from models.schema import ModelSchema
from core.engine import Engine
from core.sensitivity import SensitivityAnalysis
from core.steady_state import SteadyStateFinder
from visualization.plotly_visualizer import PlotlyVisualizer
from visualization.network_graph import NetworkGraph

def main():
    st.set_page_config(page_title="Pathophysiology Simulator Dashboard", layout="wide")
    st.title("Human Physiology and Pathology Simulator - Research Dashboard")

    # Model Loading
    st.sidebar.header("Model Configuration")

    # Load default complex model
    with open('models/example_multi_organ.json', 'r') as f:
        default_model = json.load(f)

    model_json = st.sidebar.text_area("Model JSON", json.dumps(default_model, indent=2), height=200)

    try:
        model_dict = json.loads(model_json)
        model = ModelSchema(**model_dict)
        engine = Engine(model)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulation", "Bifurcation", "Sensitivity & Stability", "Structure", "Raw Data"])

        with tab1:
            st.sidebar.subheader("Simulation Parameters")
            t_max = st.sidebar.slider("Simulation Time", 10, 2000, 1000)

            for param in model.parameters:
                new_val = st.sidebar.number_input(f"{param.name}", value=float(param.value), key=f"p_{param.name}")
                engine.set_parameter(param.name, new_val)

            if st.sidebar.button("Run Simulation"):
                history = engine.run((0, t_max))
                st.session_state['history'] = history
                st.session_state['engine'] = engine

            if 'history' in st.session_state:
                history = st.session_state['history']
                st.header("Time-Series Results")
                vars_to_plot = st.multiselect("Select Variables", history.columns[1:], default=history.columns[1:4])
                st.plotly_chart(PlotlyVisualizer.plot_time_series(history, vars_to_plot), use_container_width=True)

                st.header("Phase Space Portrait")
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis", history.columns[1:], index=0)
                with col2:
                    y_axis = st.selectbox("Y-axis", history.columns[1:], index=1)
                st.plotly_chart(PlotlyVisualizer.plot_phase_space(history, x_axis, y_axis), use_container_width=True)

        with tab2:
            if 'engine' in st.session_state:
                st.header("Bifurcation Scan")
                sa = SensitivityAnalysis(st.session_state['engine'])
                bif_param = st.selectbox("Bifurcation Parameter", [p.name for p in model.parameters])
                p_min = st.number_input("Min Value", value=0.0)
                p_max = st.number_input("Max Value", value=2.0)

                if st.button("Run Scan"):
                    results = sa.scan_bifurcation(bif_param, (p_min, p_max), n_steps=20)
                    bif_df = pd.DataFrame([{
                        'param': r['param_value'],
                        'max_eig': np.max(np.real(r['eigenvalues'])),
                        'is_stable': r['is_stable']
                    } for r in results])
                    st.line_chart(bif_df.set_index('param')['max_eig'])
                    st.write("Stability Map (Red = Unstable, Blue = Stable)")
                    # Placeholder for stability plot

        with tab3:
            if 'engine' in st.session_state:
                st.header("Local Stability & Sensitivity")
                sa = SensitivityAnalysis(st.session_state['engine'])
                t_sa = st.number_input("Analysis Timepoint", value=0.0)
                eig, stable = sa.analyze_stability(t_sa)
                st.write(f"Stable: {stable}, Eigenvalues: {eig}")

        with tab4:
            st.header("Dependency Network")
            ng = NetworkGraph(engine.validator.get_dependency_graph())
            st.plotly_chart(ng.plot_interactive_graph(), use_container_width=True)

        with tab5:
            if 'history' in st.session_state:
                st.dataframe(st.session_state['history'])

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
