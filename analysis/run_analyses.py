import json


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
            from analysis.training_analysis import TrainingAnalysis
            training_analysis = TrainingAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            training_analysis.run()
        if self.analysis_config['parameter_analysis']['enabled']:
            from analysis.parameter_analysis import ParameterAnalysis
            param_analysis = ParameterAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            param_analysis.run()
        if self.analysis_config['parameter_analysis_plot']['enabled']:
            from analysis.parameter_analysis_plot import ParameterAnalysisPlot
            param_analysis_plot = ParameterAnalysisPlot(self.analysis_config, self.run_config, self.seeds_file)
            param_analysis_plot.run()
        if self.analysis_config['sharpness_analysis']['enabled']:
            from analysis.sharpness_analysis import SharpnessAnalysis
            sharpness_analysis = SharpnessAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            sharpness_analysis.run()
        if self.analysis_config['sharpness_analysis_metrics']['enabled']:
            from analysis.sharpness_analysis_metrics import SharpnessAnalysisMetrics
            sharpness_analysis_metrics = SharpnessAnalysisMetrics(self.analysis_config, self.run_config,
                                                                  self.seeds_file)
            sharpness_analysis_metrics.run()
        if self.analysis_config['sharpness_analysis_plot']['enabled']:
            from analysis.sharpness_analysis_plot import SharpnessAnalysisPlot
            sharpness_analysis_plot = SharpnessAnalysisPlot(self.analysis_config, self.run_config, self.seeds_file)
            sharpness_analysis_plot.run()
        if self.analysis_config['sharpness_analysis_metrics_plot']['enabled']:
            from analysis.sharpness_analysis_metrics_plot import SharpnessAnalysisMetricsPlot
            sharpness_analysis_metrics_plot = SharpnessAnalysisMetricsPlot(self.analysis_config, self.run_config,
                                                                           self.seeds_file)
            sharpness_analysis_metrics_plot.run()
        if self.analysis_config['inference_analysis']['enabled']:
            from analysis.inference_analysis import InferenceAnalysis
            inference_analysis = InferenceAnalysis(self.analysis_config, self.run_config, self.seeds_file)
            inference_analysis.run()
        if self.analysis_config['correlation_plot']['enabled']:
            from analysis.correlation_plot import CorrelationPlot
            correlation_plot = CorrelationPlot(self.analysis_config, self.run_config, self.seeds_file)
            correlation_plot.run()
        if self.analysis_config['hypothesis_testing']['enabled']:
            from analysis.hypothesis_testing import HypothesisTesting
            hypothesis_testing = HypothesisTesting(self.analysis_config, self.run_config, self.seeds_file)
            hypothesis_testing.run()
