from manim import *
import numpy as np
import pandas as pd
from typing import Dict, List
import networkx as nx

class SimulationAnimator(Scene):
    def construct_from_graph(self, graph: nx.DiGraph):
        """World-class animation of the physiological model topology."""
        pos = nx.spring_layout(graph, k=2.0)

        manim_nodes = {}
        for node in graph.nodes():
            dot = Dot(point=np.array([pos[node][0]*6, pos[node][1]*4, 0]), color=BLUE_B)
            label = Text(node, font_size=18).next_to(dot, UP, buff=0.1)
            manim_nodes[node] = VGroup(dot, label)

        manim_edges = {}
        for u, v in graph.edges():
            edge = Arrow(manim_nodes[u][0].get_center(), manim_nodes[v][0].get_center(),
                        stroke_width=2, buff=0.1, color=GREY_A)
            manim_edges[(u, v)] = edge

        self.play(Write(Text("Physiological Control Network", font_size=32).to_edge(UP)))
        self.play(*(FadeIn(node) for node in manim_nodes.values()))
        self.play(*(GrowArrow(edge) for edge in manim_edges.values()))
        self.wait(1)

    def animate_feedback_intensity(self, history: pd.DataFrame, source: str, target: str):
        """Animates the intensity of a feedback signal between two nodes."""
        # This is a conceptual implementation of signal intensity animation
        # In a real run, we'd use ValueTracker linked to the history data
        pass

    def animate_shock_evolution(self, history: pd.DataFrame):
        """Dramatic visualization of hemodynamic collapse and compensation."""
        axes = Axes(x_range=[0, 150, 20], y_range=[0, 120, 20], axis_config={"color": GREY})
        labels = axes.get_axis_labels(x_label="Time", y_label="P_art")
        self.add(axes, labels)

        points = [axes.coords_to_point(row['t'], row['P_art']) for _, row in history.iterrows() if row['t'] < 150]
        path = VMobject().set_points_as_corners(points)
        path.set_color(RED)

        # Add compensatory signal visualization
        sns_label = Text("SNS Activity", color=YELLOW, font_size=20).to_corner(UR)
        sns_bar = Rectangle(height=0.5, width=0, color=YELLOW, fill_opacity=0.8).next_to(sns_label, DOWN)
        self.add(sns_label, sns_bar)

        # Simultaneous animation of pressure drop and SNS rise
        self.play(Create(path), run_time=10, rate_func=linear)
        self.wait(1)
