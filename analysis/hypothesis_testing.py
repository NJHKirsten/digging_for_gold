import os

import pandas as pd

from analysis.analysis import Analysis
from scipy.stats import pearsonr


class HypothesisTesting(Analysis):

    def run(self):
        print(f'Correlation Hyptothesis Testing')
        self.__test_all_correlations()

    def __test_all_correlations(self):
        for sharpness_config in self.analysis_config['sharpness_analysis_metrics_plot']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")

            csv_graph_path = f"parameter_analysis/{self.analysis_config['run']}.csv"
            param_graph = pd.read_csv(csv_graph_path)

            # param_graph = param_graph.melt(
            #     'seed', var_name='param_cols', value_name='param_vals')

            csv_graph_path = f"sharpness_measures/{self.analysis_config['run']}/{sharpness_config['name']}.csv"
            sharpness_graph = pd.read_csv(csv_graph_path)
            # sharpness_graph = sharpness_graph.melt(
            #     'seed', var_name='sharpness_cols', value_name='sharpness_vals')

            correlation_graph = param_graph.merge(sharpness_graph, on='seed')

            hypothesis_tests = []
            for col_name in sharpness_graph.columns:
                for row_name in param_graph.columns:
                    if (col_name == 'seed'
                            or row_name == 'seed'
                            or col_name == 'max_symmetry'
                            or col_name == 'average_symmetry'):
                        continue
                    print(f"p-{col_name}-{row_name}: {pearsonr(correlation_graph[col_name], correlation_graph[row_name])}")
                    r, p = pearsonr(correlation_graph[col_name], correlation_graph[row_name])
                    hypothesis_tests.append({
                        'sharpness_metric': col_name,
                        'regularisation_metric': row_name,
                        'r': r,
                        'p': p
                    })

            hypothesis_tests_df = pd.DataFrame(hypothesis_tests)
            os.makedirs('hypothesis_tests', exist_ok=True)
            hypothesis_tests_df.to_csv(f"hypothesis_tests/{self.analysis_config['run']}.csv", index=False)
