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


class SharpnessAnalysisMetrics(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))

        print(f'Sharpness Metrics')
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")
            self.__calculate_sharpness(seeds, sharpness_config)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __calculate_sharpness(self, seeds, sharpness_config):

        csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
        sharpness_graph = []

        for sample in range(self.run_config['sample_size']):
            seed = seeds[sample]
            sharpness_graph.append(
                self.__calculate_individual_sharpness_sample(sharpness_config, seed))

        os.makedirs(os.path.dirname(csv_graph_path), exist_ok=True)
        pd.DataFrame(sharpness_graph).to_csv(csv_graph_path, index=False)

    def __calculate_individual_sharpness_sample(self, sharpness_config, seed):

        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{seed}.csv"
        sharpness_graph = pd.read_csv(csv_graph_path)

        sharpness_graph['train_loss'] = sharpness_graph['train_loss'] / 60000
        sharpness_graph['test_loss'] = sharpness_graph['test_loss'] / 10000

        l_sub = sharpness_graph[sharpness_graph['step'] < 0]
        l_add = sharpness_graph[sharpness_graph['step'] > 0]
        l_opt = sharpness_graph[sharpness_graph['step'] == 0]

        average_sharpness = self.__average_sharpness(l_sub, l_add, l_opt)
        # average_sharpness_alt = self.__average_sharpness_alt(l_sub, l_add, l_opt)
        max_sharpness = self.__max_sharpness(l_sub, l_add, l_opt)
        max_symmetry = self.__max_symmetry(l_sub, l_add, l_opt)
        average_symmetry = self.__average_symmetry(l_sub, l_add, l_opt)

        print(f"Seed: {seed}")
        print(f"Average Sharpness: {average_sharpness}")
        # print(f"Average Sharpness Alt: {average_sharpness_alt}")
        print(f"Max Sharpness: {max_sharpness}")
        print(f"Max Symmetry: {max_symmetry}")
        print(f"Average Symmetry: {average_symmetry}")

        return {
            'seed': seed,
            'average_sharpness': average_sharpness,
            'max_sharpness': max_sharpness,
            'max_symmetry': max_symmetry,
            'average_symmetry': average_symmetry
        }

    def __average_sharpness(self, l_sub, l_add, l_opt):
        l_opt_loss = l_opt['train_loss'].max()
        l_sub_avg = (l_sub['train_loss'] - l_opt_loss).mean()
        l_add_avg = (l_add['train_loss'] - l_opt_loss).mean()
        return (l_sub_avg + l_add_avg) / 2

    def __average_sharpness_alt(self, l_sub, l_add, l_opt):
        l_opt_loss = l_opt['train_loss'].max()


        v = 0
        for k in l_add['step'].unique():
            l_sub_step = l_sub.loc[l_sub['step'] == -1*k]
            l_add_step = l_add.loc[l_add['step'] == k]
            l_sub_avg = 0
            l_add_avg = 0
            for sample in l_sub['sample'].unique():
                l_sub_avg += (l_sub_step.loc[l_sub_step['sample'] == sample]['train_loss'] - l_opt_loss).max()
                l_add_avg += (l_add_step.loc[l_add_step['sample'] == sample]['train_loss'] - l_opt_loss).max()

            v += (l_sub_avg + l_add_avg) / (2 * len(l_sub['sample'].unique()))
        return v / len(l_add['step'].unique())

    def __max_sharpness(self, l_sub, l_add, l_opt):
        l_sub_max = l_sub['train_loss'].max()
        l_add_max = l_add['train_loss'].max()
        l_opt_loss = l_opt['train_loss'].max()

        return (l_sub_max - l_opt_loss) + (l_add_max - l_opt_loss)

    def __max_symmetry(self, l_sub, l_add, l_opt):
        l_sub_max = l_sub['train_loss'].max()
        l_add_max = l_add['train_loss'].max()
        return abs(l_sub_max - l_add_max) / max(l_sub_max, l_add_max)

    def __average_symmetry(self, l_sub, l_add, l_opt):
        l_opt_loss = l_opt['train_loss'].max()
        l_sub_avg = (l_sub['train_loss'] - l_opt_loss).mean()
        l_add_avg = (l_add['train_loss'] - l_opt_loss).mean()
        return abs(l_sub_avg - l_add_avg) / max(l_sub_avg, l_add_avg)
