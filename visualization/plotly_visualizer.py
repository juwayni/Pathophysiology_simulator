import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional

class PlotlyVisualizer:
    @staticmethod
    def plot_time_series(df: pd.DataFrame, variables: List[str], title: str = "Simulation Results") -> go.Figure:
        fig = go.Figure()
        for var in variables:
            fig.add_trace(go.Scatter(x=df['t'], y=df[var], mode='lines', name=var))

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value", template="plotly_white")
        return fig

    @staticmethod
    def plot_phase_space(df: pd.DataFrame, x_var: str, y_var: str, z_var: Optional[str] = None) -> go.Figure:
        if z_var:
            fig = go.Figure(data=[go.Scatter3d(x=df[x_var], y=df[y_var], z=df[z_var], mode='lines')])
            fig.update_layout(scene=dict(xaxis_title=x_var, yaxis_title=y_var, zaxis_title=z_var))
        else:
            fig = go.Figure(data=[go.Scatter(x=df[x_var], y=df[y_var], mode='lines')])
            fig.update_layout(xaxis_title=x_var, yaxis_title=y_var)

        fig.update_layout(title=f"Phase Portrait: {x_var} vs {y_var} {'vs ' + z_var if z_var else ''}")
        return fig
