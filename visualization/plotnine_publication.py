from plotnine import (ggplot, aes, geom_line, theme_minimal, labs,
                        facet_wrap, theme, element_text, geom_point)
import pandas as pd
from typing import List

class PlotninePublication:
    @staticmethod
    def plot_time_series(df: pd.DataFrame, variables: List[str], title: str = "Simulation Results") -> ggplot:
        """Generates a publication-quality faceted plot of time-series results."""
        df_melted = df.melt(id_vars=['t'], value_vars=variables, var_name='Variable', value_name='Value')

        p = (ggplot(df_melted, aes(x='t', y='Value', color='Variable'))
             + geom_line(size=1.2)
             + theme_minimal()
             + theme(
                 text=element_text(family='sans-serif'),
                 title=element_text(size=16, face='bold'),
                 axis_title=element_text(size=12),
                 legend_position='none',
                 strip_text=element_text(size=12, face='bold')
             )
             + labs(title=title, x="Time", y="Value")
             + facet_wrap('~Variable', scales='free_y'))

        return p

    @staticmethod
    def plot_phase_portrait(df: pd.DataFrame, x: str, y: str, title: str = "Phase Portrait") -> ggplot:
        """Generates a high-quality phase portrait plot."""
        p = (ggplot(df, aes(x=x, y=y))
             + geom_line(size=0.8, alpha=0.6)
             + geom_point(df.head(1), aes(x=x, y=y), color='green', size=3) # Start point
             + geom_point(df.tail(1), aes(x=x, y=y), color='red', size=3)   # End point
             + theme_minimal()
             + labs(title=title, x=x, y=y))

        return p
