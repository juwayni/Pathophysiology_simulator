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
    st.set_page_config(page_title="Pathophysiology Research Platform", layout="wide")
    st.title("Advanced Human Physiology and Pathology Simulator")

    # Model Catalog
    models_available = {
        "Hemorrhagic Shock": "models/example_shock.json",
        "Multi-Organ Digital Twin": "models/example_multi_organ.json",
        "Basic Cardiovascular": "models/example_cardiovascular.json"
    }

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Choose Scenario", list(models_available.keys()))

    with open(models_available[model_choice], 'r') as f:
        default_model = json.load(f)

    model_json = st.sidebar.text_area("Model Specification (JSON)", json.dumps(default_model, indent=2), height=200)

    try:
        model_dict = json.loads(model_json)
        model = ModelSchema(**model_dict)
        engine = Engine(model)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Simulation & Events",
            "🔍 Sensitivity Analytics",
            "⚖️ Stability & Bifurcation",
            "🕸️ Model Topology",
            "📋 Export Data"
        ])

        with tab1:
            col_cfg, col_res = st.columns([1, 3])
            with col_cfg:
                st.subheader("Control Panel")
                t_max = st.slider("Horizon", 10, 2000, 500)

                st.write("**Real-time Parameters**")
                for param in model.parameters:
                    new_val = st.number_input(f"{param.name}", value=float(param.value), key=f"p_{param.name}")
                    engine.set_parameter(param.name, new_val)

                run_btn = st.button("🚀 Run Experiment", use_container_width=True)

            with col_res:
                if run_btn:
                    history = engine.run((0, t_max))
                    st.session_state['history'] = history
                    st.session_state['engine'] = engine

                if 'history' in st.session_state:
                    history = st.session_state['history']
                    st.header("Physiological Time-Series")
                    vars_to_plot = st.multiselect("Active Channels", history.columns[1:], default=history.columns[1:4])
                    st.plotly_chart(PlotlyVisualizer.plot_time_series(history, vars_to_plot), use_container_width=True)

                    st.header("Intervention Timeline")
                    if model.events:
                        event_data = []
                        # Simplification: show event triggers
                        for e in model.events:
                             event_data.append({"Event": e.name, "Condition": e.condition, "Action": e.action})
                        st.table(pd.DataFrame(event_data))

        with tab2:
            if 'engine' in st.session_state:
                st.header("Normalized Sensitivity Ranking")
                sa = SensitivityAnalysis(st.session_state['engine'])
                ranking = sa.get_sensitivity_ranking(0.0)

                fig_rank = px.bar(ranking, x='influence', y='parameter', orientation='h',
                                 title="Parameter Influence Ranking (Normalized)",
                                 color='influence', color_continuous_scale='Viridis')
                st.plotly_chart(fig_rank, use_container_width=True)

                st.header("Sensitivity Matrix Heatmap")
                sens_matrix = sa.compute_local_sensitivity(0.0)
                st.plotly_chart(px.imshow(sens_matrix,
                                        labels=dict(x="Parameters", y="States", color="Sensitivity"),
                                        x=[p.name for p in model.parameters],
                                        y=[v.name for v in model.variables if v.type == 'state']),
                                use_container_width=True)
            else:
                st.warning("Run a simulation to unlock analytics.")

        with tab3:
            if 'engine' in st.session_state:
                st.header("Stability & Bifurcation Scan")
                sa = SensitivityAnalysis(st.session_state['engine'])
                bif_param = st.selectbox("Control Parameter", [p.name for p in model.parameters])
                p_min = st.number_input("Scan Min", value=0.0)
                p_max = st.number_input("Scan Max", value=2.0)

                if st.button("🔬 Perform Scan"):
                    results = sa.scan_bifurcation(bif_param, (p_min, p_max), n_steps=30)
                    bif_df = pd.DataFrame([{
                        'param': r['param_value'],
                        'stability_index': np.max(np.real(r['eigenvalues'])),
                        'status': 'Stable' if r['is_stable'] else 'Unstable'
                    } for r in results])

                    fig_bif = px.line(bif_df, x='param', y='stability_index', color='status',
                                    title=f"Bifurcation Map for {bif_param}",
                                    labels={'stability_index': 'Max Real Eigenvalue (Decay Rate)'})
                    st.plotly_chart(fig_bif, use_container_width=True)

        with tab4:
            st.header("Dynamical Dependency Network")
            ng = NetworkGraph(engine.validator.get_dependency_graph())
            st.plotly_chart(ng.plot_interactive_graph(), use_container_width=True)
            st.info("Nodes represent physiological variables; edges represent causal mathematical dependencies.")

        with tab5:
            if 'history' in st.session_state:
                st.download_button("📥 Download Results (CSV)",
                                 st.session_state['history'].to_csv(index=False),
                                 "simulation_results.csv", "text/csv")
                st.dataframe(st.session_state['history'])

    except Exception as e:
        st.error(f"Engine Fault: {e}")

if __name__ == "__main__":
    main()
