import networkx as nx
import plotly.graph_objects as go
from typing import Dict

class NetworkGraph:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_interactive_graph(self) -> go.Figure:
        pos = nx.spring_layout(self.graph)

        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(self.graph.nodes()), hoverinfo='text',
                                 marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], line_width=2),
                                 textposition="top center")

        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest',
                                                                         margin=dict(b=20, l=5, r=5, t=40),
                                                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig
