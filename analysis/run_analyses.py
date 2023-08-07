import json
import os

import pandas as pd

from analysis.inference_analysis import InferenceAnalysis
from analysis.sharpness_analysis import SharpnessAnalysis
from analysis.sharpness_analysis_plot import SharpnessAnalysisPlot
from analysis.training_analysis import TrainingAnalysis
from analysis.weight_analysis import WeightAnalysis


class RunAnalyses:

    def __init__(self, anaysis_config, run_config, seeds_file="seeds.json"):
        self.analysis_config = anaysis_config
        self.run_config = run_config
        self.seeds_file = seeds_file

    @staticmethod
    def run(analysis_config_name, seeds_file="seeds.json"):
        analysis_config = json.load(open(f'analysis_configs/{analysis_config_name}.json', 'r'))
        for run in analysis_config['runs']:
            print(f"Running analysis for run {run}")
            run_config = json.load(open(f'run_configs/{run}.json', 'r'))
            analysis_config['run'] = run
            analysis = RunAnalyses(analysis_config, run_config, seeds_file)
            analysis.run_all()

    def run_all(self):
        if self.analysis_config['training_analysis']['enabled']:
            training_analysis = TrainingAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            training_analysis.run()
        if self.analysis_config['weight_analysis']['enabled']:
            weight_analysis = WeightAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            weight_analysis.run()
        if self.analysis_config['sharpness_analysis']['enabled']:
            sharpness_analysis = SharpnessAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            sharpness_analysis.run()
        if self.analysis_config['sharpness_analysis_plot']['enabled']:
            sharpness_analysis = SharpnessAnalysisPlot(self.analysis_config, self.run_config, self.seeds_file)
            sharpness_analysis.run()
        if self.analysis_config['inference_analysis']['enabled']:
            inference_analysis = InferenceAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            inference_analysis.run()

