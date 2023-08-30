import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.analysis import Analysis


class ParameterAnalysisPlot(Analysis):

    def run(self):
        print(f'Parameter Metrics Plots')
        self.__plot_metrics()

    def __plot_metrics(self):
        csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
        param_graph = pd.read_csv(csv_graph_path)

        self.__plot_all(param_graph)

    def __plot_all(self, param_graph):
        param_graph = param_graph.melt(
            'seed', var_name='cols', value_name='vals')

        g = sns.displot(x='vals',
                        col='cols',
                        data=param_graph,
                        col_wrap=4,
                        common_bins=False,
                        facet_kws={'sharex': False, 'sharey': False})
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Value", "Count")
        plt.show()
