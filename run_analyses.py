import json
import os

import pandas as pd

from analysis.sharpness_analysis import SharpnessAnalysis
from analysis.training_analysis import TrainingAnalysis
from analysis.weight_analysis import WeightAnalysis


class RunAnalyses:

    def __init__(self, run_config, seeds_file="seeds.json"):
        self.run_config = run_config
        self.seeds_file = seeds_file

    @staticmethod
    def run(run_name, seeds_file="seeds.json"):
        run_config = json.load(open(f'run_configs/{run_name}.json', 'r'))
        run_config['run'] = run_name
        analysis = RunAnalyses(run_config, seeds_file)
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
        model_name, model_class, dataset_name, dataset_class, sample_size = \
            (self.run_config["model_name"],
             self.run_config["model_class"],
             self.run_config["dataset_name"],
             self.run_config["dataset_class"],
             self.run_config["sample_size"])
        training_analysis = TrainingAnalysis(model_name,
                                             model_class,
                                             dataset_name,
                                             dataset_class,
                                             sample_size,
                                             self.run_config)
        training_analysis.run()
        weight_analysis = WeightAnalysis(model_name,
                                         model_class,
                                         dataset_name,
                                         dataset_class,
                                         sample_size,
                                         self.run_config)
        weight_analysis.run()
        sharpness_analysis = SharpnessAnalysis(model_name,
                                               model_class,
                                               dataset_name,
                                               dataset_class,
                                               sample_size,
                                               self.run_config)
        sharpness_analysis.run()
