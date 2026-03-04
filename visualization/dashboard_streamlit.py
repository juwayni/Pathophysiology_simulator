import streamlit as st
import numpy as np
import json
import pandas as pd
import plotly.express as px
from models.schema import ModelSchema
from core.engine import Engine
from core.sensitivity import SensitivityAnalysis
from visualization.plotly_visualizer import PlotlyVisualizer
from visualization.network_graph import NetworkGraph

def main():
    st.set_page_config(page_title="Pathophysiology Research Studio", layout="wide")
    st.title("Advanced Human Physiology and Pathology Research Studio")

    models_available = {
        "Hemorrhagic Shock": "models/example_shock.json",
        "Multi-Organ Digital Twin": "models/example_multi_organ.json",
        "Basic Cardiovascular": "models/example_cardiovascular.json"
    }

    st.sidebar.header("Experimental Setup")
    model_choice = st.sidebar.selectbox("Model Template", list(models_available.keys()))

    # Clinical Presets
    presets = {
        "Baseline (Normal)": {},
        "Autonomic Failure (Low K_baro)": {"K_baro": 0.01},
        "Acute Dehydration (Low V_total)": {"V_total": 3500.0},
        "Salt-Sensitive Hypertension": {"K_renal": 0.01, "Na_intake": 0.5}
    }
    st.sidebar.subheader("Clinical Presets")
    preset_choice = st.sidebar.selectbox("Apply Condition", list(presets.keys()))

    with open(models_available[model_choice], 'r') as f:
        model_dict = json.load(f)

    # Apply preset overrides
    for k, v in presets[preset_choice].items():
         # Override variables or parameters
         for var in model_dict['variables']:
             if var['name'] == k: var['initial_value'] = v
         for param in model_dict['parameters']:
             if param['name'] == k: param['value'] = v

    model_json = st.sidebar.text_area("Live Model Definition", json.dumps(model_dict, indent=2), height=200)

    try:
        model = ModelSchema(**json.loads(model_json))
        engine = Engine(model)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🖥️ Virtual Experiment", "📊 Analytic Suite", "🧪 Topology & Stability", "💾 Export"
        ])

        with tab1:
            col_in, col_out = st.columns([1, 2])
            with col_in:
                st.subheader("Intervention Controls")
                t_max = st.slider("Simulation Time", 10, 2000, 500)
                if st.button("🚀 Execute Simulation", use_container_width=True):
                    st.session_state['history'] = engine.run((0, t_max))
                    st.session_state['engine'] = engine

            with col_out:
                if 'history' in st.session_state:
                    history = st.session_state['history']
                    st.plotly_chart(PlotlyVisualizer.plot_time_series(history, history.columns[1:5]), use_container_width=True)
                    st.write("**Event Log**")
                    st.dataframe(pd.DataFrame([{"Event": e.name, "Trigger": e.condition} for e in model.events]))

        with tab2:
            if 'engine' in st.session_state:
                st.header("Sensitivity Intelligence")
                sa = SensitivityAnalysis(st.session_state['engine'])
                ranking = sa.get_sensitivity_ranking(0.0)
                st.plotly_chart(px.bar(ranking, x='influence', y='parameter', orientation='h', title="Parameter Impact"), use_container_width=True)

                st.header("Jacobian Sensitivity Matrix")
                sens_matrix = sa.compute_local_sensitivity(0.0)
                st.plotly_chart(px.imshow(sens_matrix, x=[p.name for p in model.parameters], y=[v.name for v in model.variables if v.type == 'state']), use_container_width=True)
            else:
                st.info("Run an experiment to view analytics.")

        with tab3:
            st.header("Dynamical System Stability")
            if 'engine' in st.session_state:
                sa = SensitivityAnalysis(st.session_state['engine'])
                eig, stable = sa.analyze_stability(0.0)
                st.metric("Local Stability", "Asymptotically Stable" if stable else "Unstable")
                st.write(f"Eigenvalues: {eig}")

            st.header("Topological Connectivity")
            ng = NetworkGraph(engine.validator.get_dependency_graph())
            st.plotly_chart(ng.plot_interactive_graph(), use_container_width=True)

        with tab4:
             if 'history' in st.session_state:
                 st.dataframe(st.session_state['history'])

    except Exception as e:
        st.error(f"Configuration Error: {e}")

if __name__ == "__main__":
    main()
