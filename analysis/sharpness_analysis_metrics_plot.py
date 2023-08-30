import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.analysis import Analysis


class SharpnessAnalysisMetricsPlot(Analysis):

    def run(self):

        print(f'Sharpness Metric Plot')
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")
            self.__plot_metrics(sharpness_config)

    def __plot_metrics(self, sharpness_config):
        csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
        sharpness_graph = pd.read_csv(csv_graph_path)

        self.__plot_average_sharpness(sharpness_graph, sharpness_config['name'])
        self.__plot_max_sharpness(sharpness_graph, sharpness_config['name'])
        self.__plot_average_symmetry(sharpness_graph, sharpness_config['name'])
        self.__plot_max_symmetry(sharpness_graph, sharpness_config['name'])

    def __plot_average_sharpness(self, sharpness_graph, config_name):
        ax = sns.displot(x='average_sharpness',
                         data=sharpness_graph)

        ax.fig.suptitle(f"Average sharpness {config_name}\n(Run: {self.analysis_config['run']})")
        ax.set_xlabels("Average sharpness")
        ax.set_ylabels("Count")

        ax.fig.subplots_adjust(top=0.9)

        plt.show()

    def __plot_max_sharpness(self, sharpness_graph, config_name):
        ax = sns.displot(x='max_sharpness',
                         data=sharpness_graph)

        ax.fig.suptitle(f"Max sharpness {config_name}\n(Run: {self.analysis_config['run']})")
        ax.set_xlabels("Max sharpness")
        ax.set_ylabels("Count")

        ax.fig.subplots_adjust(top=0.9)

        plt.show()

    def __plot_average_symmetry(self, sharpness_graph, config_name):
        ax = sns.displot(x='average_symmetry',
                         data=sharpness_graph)

        ax.fig.suptitle(f"Average symmetry {config_name}\n(Run: {self.analysis_config['run']})")
        ax.set_xlabels("Average symmetry")
        ax.set_ylabels("Count")

        ax.fig.subplots_adjust(top=0.9)

        plt.show()

    def __plot_max_symmetry(self, sharpness_graph, config_name):
        ax = sns.displot(x='max_symmetry',
                         data=sharpness_graph)

        ax.fig.suptitle(f"Max symmetry {config_name}\n(Run: {self.analysis_config['run']})")
        ax.set_xlabels("Max symmetry")
        ax.set_ylabels("Count")

        ax.fig.subplots_adjust(top=0.9)

        plt.show()
