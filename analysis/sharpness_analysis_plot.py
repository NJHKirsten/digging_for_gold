import copy
import json
import os
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.analysis import Analysis
from model_imports import *
from dataset_imports import *


class SharpnessAnalysisPlot(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))

        print(f'Sharpness Plot')
        for sharpness_config in self.analysis_config['sharpness_analysis']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")
            if self.analysis_config['sharpness_analysis_plot']['individual_plots']:
                self.__plot_individual_sharpness(seeds, sharpness_config)
            else:
                self.__plot_sharpness(seeds, sharpness_config)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __plot_sharpness(self, seeds, sharpness_config):
        # TODO
        pass

    def __plot_individual_sharpness(self, seeds, sharpness_config):

        for seed in seeds:
            self.__plot_individual_sharpness_sample(sharpness_config, seed)

    def __plot_individual_sharpness_sample(self, sharpness_config, seed):

        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{seed}.csv"
        sharpness_graph = pd.read_csv(csv_graph_path)

        ax = sns.lineplot(x='step',
                          y='train_loss',
                          hue='sample',
                          data=sharpness_graph)

        ax.set_title(f"Sharpness {sharpness_config['name']} - {seed}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")

        plt.show()
