from plotnine import ggplot, aes, geom_line, theme_minimal, labs, facet_wrap
import pandas as pd
from typing import List

class PlotninePublication:
    @staticmethod
    def plot_time_series(df: pd.DataFrame, variables: List[str], title: str = "Simulation Results") -> ggplot:
        # Melt the dataframe for plotting
        df_melted = df.melt(id_vars=['t'], value_vars=variables, var_name='Variable', value_name='Value')

        p = (ggplot(df_melted, aes(x='t', y='Value', color='Variable'))
             + geom_line(size=1)
             + theme_minimal()
             + labs(title=title, x="Time", y="Value")
             + facet_wrap('~Variable', scales='free_y'))

        return p
