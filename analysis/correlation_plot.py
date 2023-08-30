import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.analysis import Analysis


class CorrelationPlot(Analysis):

    def run(self):
        print(f'Correlation Plots')
        # self.__plot_walk_metrics_vs_param_dist()
        # self.__plot_walk_metrics_vs_param_scatter()
        # self.__plot_walk_metrics_vs_param_scatter_with_symmetry()
        self.__plot_walk_metrics_vs_param_scatter_without_symmetry()
        # self.__001_vs_01_sharpness()

    def __plot_walk_metrics_vs_param_dist(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")

            csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
            param_graph = pd.read_csv(csv_graph_path)

            param_graph = param_graph.melt(
                'seed', var_name='param_cols', value_name='param_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
            sharpness_graph = pd.read_csv(csv_graph_path)
            sharpness_graph = sharpness_graph.melt(
                'seed', var_name='sharpness_cols', value_name='sharpness_vals')

            correlation_graph = param_graph.merge(sharpness_graph, on='seed')

            g = sns.displot(x='param_vals',
                            y='sharpness_vals',
                            col='param_cols',
                            row='sharpness_cols',
                            data=correlation_graph,
                            common_bins=False,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}", row_template='{row_name}')
            g.set_axis_labels("Sharpness", "Parameters")
            plt.show()

    def __plot_walk_metrics_vs_param_scatter(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")

            csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
            param_graph = pd.read_csv(csv_graph_path)

            param_graph = param_graph.melt(
                'seed', var_name='param_cols', value_name='param_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
            sharpness_graph = pd.read_csv(csv_graph_path)
            sharpness_graph = sharpness_graph.melt(
                'seed', var_name='sharpness_cols', value_name='sharpness_vals')

            correlation_graph = param_graph.merge(sharpness_graph, on='seed')

            g = sns.relplot(x='sharpness_vals',
                            y='param_vals',
                            col='param_cols',
                            row='sharpness_cols',
                            data=correlation_graph,
                            # common_bins=False,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}", row_template='{row_name}')
            g.set_axis_labels("Parameters", "Sharpness")
            plt.show()

    def __plot_walk_metrics_vs_param_scatter_with_symmetry(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")

            csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
            param_graph = pd.read_csv(csv_graph_path)

            param_graph = param_graph.melt(
                'seed', var_name='param_cols', value_name='param_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
            sharpness_graph = pd.read_csv(csv_graph_path)
            sharpness_graph = sharpness_graph.melt(
                ['seed', 'max_symmetry', 'average_symmetry'],
                var_name='sharpness_cols',
                value_name='sharpness_vals')

            correlation_graph = param_graph.merge(sharpness_graph, on='seed')

            # color_map = plt.cm.ScalarMappable()

            g = sns.relplot(x='param_vals',
                            y='sharpness_vals',
                            col='param_cols',
                            row='sharpness_cols',
                            hue='max_symmetry',
                            palette=sns.color_palette("viridis", as_cmap=True),
                            data=correlation_graph,
                            # common_bins=False,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}", row_template='{row_name}')
            g.set_axis_labels("Sharpness", "Parameters")
            plt.show()

            g = sns.relplot(x='param_vals',
                            y='sharpness_vals',
                            col='param_cols',
                            row='sharpness_cols',
                            hue='average_symmetry',
                            palette=sns.color_palette("Spectral", as_cmap=True),
                            data=correlation_graph,
                            # common_bins=False,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}")
            g.set_axis_labels("Parameters", "Sharpness")
            plt.show()

    def __plot_walk_metrics_vs_param_scatter_without_symmetry(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")

            csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
            param_graph = pd.read_csv(csv_graph_path)

            param_graph = param_graph.melt(
                'seed', var_name='param_cols', value_name='param_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
            sharpness_graph = pd.read_csv(csv_graph_path)
            sharpness_graph = sharpness_graph.melt(
                ['seed', 'max_symmetry', 'average_symmetry'],
                var_name='sharpness_cols',
                value_name='sharpness_vals')

            correlation_graph = param_graph.merge(sharpness_graph, on='seed')

            g = sns.relplot(x='param_vals',
                            y='sharpness_vals',
                            col='param_cols',
                            row='sharpness_cols',
                            data=correlation_graph,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}", row_template='{row_name}')
            g.set_axis_labels("Parameters", "Sharpness")
            plt.show()

    def __001_vs_01_sharpness(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:


            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/0_01.csv"
            sharpness_01_graph = pd.read_csv(csv_graph_path)

            sharpness_01_graph = sharpness_01_graph.melt(
                ['seed'],
                var_name='0_01_cols',
                value_name='0_01_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/0_001.csv"
            sharpness_001_graph = pd.read_csv(csv_graph_path)

            sharpness_001_graph = sharpness_001_graph.melt(
                ['seed'],
                var_name='0_001_cols',
                value_name='0_001_vals')

            correlation_graph = sharpness_01_graph.merge(sharpness_001_graph, on='seed')

            g = sns.relplot(x='0_01_vals',
                            y='0_001_vals',
                            col='0_01_cols',
                            row='0_001_cols',
                            data=correlation_graph,
                            facet_kws={'sharex': False, 'sharey': False})
            g.set_titles(col_template="{col_name}", row_template='{row_name}')
            g.set_axis_labels("0_01", "0_001")
            plt.show()
