from manim import *
import numpy as np
import pandas as pd
from typing import Dict, List
import networkx as nx

class SimulationAnimator(Scene):
    def construct_from_graph(self, graph: nx.DiGraph):
        """World-class animation of the physiological model topology."""
        pos = nx.spring_layout(graph, k=2.0)

        # Manim representations
        manim_nodes = {}
        for node in graph.nodes():
            dot = Dot(point=np.array([pos[node][0]*6, pos[node][1]*4, 0]), color=BLUE_B)
            label = Text(node, font_size=18).next_to(dot, UP, buff=0.1)
            manim_nodes[node] = VGroup(dot, label)

        manim_edges = []
        for u, v in graph.edges():
            edge = Arrow(manim_nodes[u][0].get_center(), manim_nodes[v][0].get_center(),
                        stroke_width=2, buff=0.1, color=GREY_A)
            manim_edges.append(edge)

        # Dramatic intro
        self.play(Write(Text("Physiological Dependency Network", font_size=32).to_edge(UP)))
        self.play(*(FadeIn(node) for node in manim_nodes.values()), run_time=2)
        self.play(*(GrowArrow(edge) for edge in manim_edges), run_time=2)

        # Highlight feedback loops
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                highlight = Text("Feedback Loops Detected", color=YELLOW, font_size=24).to_edge(DOWN)
                self.play(Write(highlight))
                for cycle in cycles:
                    # Highlight edges in the cycle
                    cycle_edges = []
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i+1)%len(cycle)]
                        edge = Arrow(manim_nodes[u][0].get_center(), manim_nodes[v][0].get_center(), color=YELLOW)
                        cycle_edges.append(edge)
                    self.play(*(Create(e) for e in cycle_edges), run_time=1)
                    self.wait(0.5)
                    self.play(*(Uncreate(e) for e in cycle_edges))
        except:
            pass

        self.wait(2)

    def animate_shock_event(self, history: pd.DataFrame):
        """Specialized animation for critical events like shock."""
        title = Text("Hemorrhagic Shock Event", color=RED).to_edge(UP)
        self.add(title)

        axes = Axes(x_range=[0, 100], y_range=[0, 120], axis_config={"color": GREY})
        p_art_label = axes.get_graph_label(Text("P_art", font_size=20), x_val=90)
        self.add(axes)

        # Sample points around event
        points = [axes.coords_to_point(row['t'], row['P_art']) for _, row in history.iterrows() if row['t'] < 100]
        path = VMobject().set_points_as_corners(points)

        dot = Dot(color=RED).move_to(path.get_start())
        self.add(dot)
        self.play(MoveAlongPath(dot, path), Create(path), run_time=5, rate_func=linear)
        self.wait(1)
