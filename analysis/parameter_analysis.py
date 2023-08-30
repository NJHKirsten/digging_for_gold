import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from analysis.analysis import Analysis
from model_imports import *


class ParameterAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))
        model = self.__class_from_string(self.run_config['model_class'])()

        csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
        param_graph = []

        print(f'Parameter Analysis')
        for sample in range(self.run_config['sample_size']):
            seed = seeds[sample]
            param_graph.append(
                self.__calculate_sample_param_measures(model, seed))

        os.makedirs(os.path.dirname(csv_graph_path), exist_ok=True)
        pd.DataFrame(param_graph).to_csv(csv_graph_path, index=False)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __calculate_sample_param_measures(self, model, seed):
        model.load_state_dict(
            torch.load(
                f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seed}/original.pt"),
            strict=False)
        model.eval()

        params = torch.empty(0)

        for name, param in model.named_parameters():
            if param.requires_grad:
                params = torch.cat((params, param.flatten()))
        # for layer in model.children():
        #     params = torch.cat((params, layer.weight.flatten()))
        #     params = torch.cat((params, layer.bias.flatten()))

        params = params.detach().abs().numpy()
        if seed == 29055:
            self.__plot_param_distribution(params, seed)
        return {
            'seed': seed,
            'params_mean': params.mean(),
            'params_std': params.std(),
            'params_median': np.median(params),
            'params_percentile_5': np.percentile(params, 5),
            'params_percentile_10': np.percentile(params, 10),
            'params_percentile_20': np.percentile(params, 20),
            'params_percentile_50': np.percentile(params, 50),
            'params_below_1': (params < 0.1).sum()/params.size,
            'params_below_01': (params < 0.01).sum()/params.size,
            'params_below_001': (params < 0.001).sum()/params.size,
            'params_below_0001': (params < 0.0001).sum()/params.size,
        }

    def __plot_param_distribution(self, params, seed):
        ax = sns.displot(params)

        ax.fig.suptitle(f'Parameter Distribution - {seed}')
        ax.set_xlabels('Parameter Value')
        ax.set_ylabels('Frequency')
        ax.fig.subplots_adjust(top=0.9)
        plt.show()
