import json
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.analysis import Analysis


class SharpnessAnalysisPlot(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))

        print(f'Sharpness Plot')
        for sharpness_config in self.analysis_config['sharpness_analysis_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")
            if self.analysis_config['sharpness_analysis_plot']['individual_plots']:
                self.__plot_individual_sharpness(seeds, sharpness_config)
            else:
                self.__plot_sharpness(seeds, sharpness_config)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __plot_sharpness(self, seeds, sharpness_config):
        sharpness_graph = self.__get_processed_sharpness_data(seeds, sharpness_config)
        self.__plot_all_sharpness(sharpness_graph, 'Sharpness of attraction basins')
        if self.analysis_config['sharpness_analysis_plot']['plot_max_and_min']:
            max_sharpness_graph, min_sharpness_graph = self.__get_max_and_min_sharpness(sharpness_config)
            self.__plot_all_sharpness(max_sharpness_graph, 'Sharpest minimum')
            self.__plot_all_sharpness(min_sharpness_graph, 'Flattest minimum')

    def __plot_all_sharpness(self, sharpness_graph, title):
        ax = sns.boxplot(x='step',
                         y='train_loss',
                         data=sharpness_graph)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        plt.show()

    def __get_processed_sharpness_data(self, seeds, sharpness_config):
        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/"
        samples = []
        for sample in range(self.run_config['sample_size']):
            seed = seeds[sample]
            samples.append(pd.read_csv(f"{csv_graph_path}/{seed}.csv"))

        # sharpness_data = (pd.concat(samples)
        #                   .drop('sample', axis=1)
        #                   .drop('test_loss', axis=1)
        #                   .groupby('step'))
        # deviations = sharpness_data.std().rename(columns={'train_loss': 'train_loss_std',
        #                                                   'test_loss': 'test_loss_std'})
        # means = sharpness_data.mean()
        # processed_sharpness_data = pd.merge(means, deviations, on='step')
        # return processed_sharpness_data
        return pd.concat(samples)

    def __plot_individual_sharpness(self, seeds, sharpness_config):

        for sample in range(self.run_config['sample_size']):
            seed = seeds[sample]
            self.__plot_individual_sharpness_sample(sharpness_config, seed)

    def __plot_individual_sharpness_sample(self, sharpness_config, seed):

        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{seed}.csv"
        sharpness_graph = pd.read_csv(csv_graph_path)

        ax = sns.lineplot(x='step',
                          y='train_loss',
                          hue='sample',
                          palette=sns.color_palette(),
                          data=sharpness_graph)

        ax.set_title(f"Sharpness {sharpness_config['name']} - {seed} (Run: {self.analysis_config['run']})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")

        plt.show()

    def __get_max_and_min_sharpness(self, sharpness_config):
        csv_sharpness_measures_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
        sharpness_measures = pd.read_csv(csv_sharpness_measures_path)
        max_seed = sharpness_measures.loc[sharpness_measures['average_sharpness'].idxmax(), 'seed']
        min_seed = sharpness_measures.loc[sharpness_measures['average_sharpness'].idxmin(), 'seed']
        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{max_seed}.csv"
        max_sharpness = pd.read_csv(csv_graph_path)
        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{min_seed}.csv"
        min_sharpness = pd.read_csv(csv_graph_path)
        return max_sharpness, min_sharpness
