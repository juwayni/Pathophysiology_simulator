from manim import *
import numpy as np
import pandas as pd
from typing import Dict, List
import networkx as nx

class SimulationAnimator(Scene):
    def construct_from_graph(self, graph: nx.DiGraph):
        """Builds an animation based on the model dependency graph."""
        pos = nx.spring_layout(graph)

        # Create Manim nodes and edges
        manim_nodes = {}
        for node in graph.nodes():
            dot = Dot(point=np.array([pos[node][0]*5, pos[node][1]*3, 0]))
            label = Text(node, font_size=18).next_to(dot, UP)
            manim_nodes[node] = VGroup(dot, label)

        manim_edges = []
        for u, v in graph.edges():
            edge = Arrow(manim_nodes[u].get_center(), manim_nodes[v].get_center(), stroke_width=2, buff=0.1)
            manim_edges.append(edge)

        # Animate construction
        self.play(*(Create(node) for node in manim_nodes.values()))
        self.play(*(Create(edge) for edge in manim_edges))
        self.wait(2)

    def animate_time_series(self, history: pd.DataFrame, variables: List[str]):
        """Animates a point moving along the time-series curves."""
        axes = Axes(
            x_range=[0, history['t'].max(), history['t'].max()/5],
            y_range=[0, history[variables].max().max() * 1.1, history[variables].max().max()/4],
            axis_config={"color": GREY}
        )
        self.add(axes)

        for var in variables:
            points = [axes.coords_to_point(row['t'], row[var]) for _, row in history.iterrows()]
            path = VMobject().set_points_as_corners(points)
            self.play(Create(path), run_time=2)
        self.wait(1)
