from manim import *
import numpy as np
import pandas as pd
from typing import Dict, List

class SimulationAnimator(Scene):
    """Base Manim class for model animation."""

    def construct_from_data(self, history: pd.DataFrame, variables: List[str]):
        # Simple animation of time-series plots
        axes = Axes(
            x_range=[0, history['t'].max(), history['t'].max()/10],
            y_range=[0, history[variables].max().max() * 1.1, history[variables].max().max()/5],
            axis_config={"color": BLUE}
        )

        labels = axes.get_axis_labels(x_label="t", y_label="Value")
        self.play(Create(axes), Write(labels))

        for var in variables:
            # Create a value tracker or use the history data directly
            # For simplicity, we'll draw the full line or animate a point
            path = VMobject()
            points = [axes.coords_to_point(row['t'], row[var]) for _, row in history.iterrows()]
            path.set_points_as_corners(points)

            label = Text(var, font_size=24).next_to(path.get_end(), RIGHT)
            self.play(Create(path), Write(label), run_time=3)

        self.wait(2)
