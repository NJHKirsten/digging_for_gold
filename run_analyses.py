import json
import os

import pandas as pd

from analysis.sharpness_analysis import SharpnessAnalysis
from analysis.training_analysis import TrainingAnalysis
from analysis.weight_analysis import WeightAnalysis


class RunAnalyses:

    def __init__(self, analysis_configs, global_config):
        self.analysis_configs = analysis_configs
        self.global_config = global_config
        self.global_config['run'] = RunAnalyses.__get_run_number()

    @staticmethod
    def run(experiments_file="experiment_config.csv",
            global_config_file="all_experiments_config.json"):
        analysis_configs = pd.read_csv(f'setups/{experiments_file}')
        global_config = json.load(open(f'config/{global_config_file}'))
        analysis = RunAnalyses(analysis_configs, global_config)
        analysis.run_all()

    @staticmethod
    def __get_run_number():
        if not os.path.exists('runs'):
            raise Exception("No runs found")
        existing_runs = os.listdir('runs')
        if len(existing_runs) == 0:
            raise Exception("No runs found")
        existing_runs.sort(reverse=True)
        return int(existing_runs[0])

    def run_all(self):
        for model_name, model_class, dataset_name, dataset_class, sample_size in zip(
                self.analysis_configs["model_name"],
                self.analysis_configs["model_class"],
                self.analysis_configs["dataset_name"],
                self.analysis_configs["dataset_class"],
                self.analysis_configs["sample_size"]):
            training_analysis = TrainingAnalysis(model_name,
                                                 model_class,
                                                 dataset_name,
                                                 dataset_class,
                                                 sample_size,
                                                 self.global_config)
            training_analysis.run()
            weight_analysis = WeightAnalysis(model_name,
                                             model_class,
                                             dataset_name,
                                             dataset_class,
                                             sample_size,
                                             self.global_config)
            weight_analysis.run()
            sharpness_analysis = SharpnessAnalysis(model_name,
                                                   model_class,
                                                   dataset_name,
                                                   dataset_class,
                                                   sample_size,
                                                   self.global_config)
            sharpness_analysis.run()
