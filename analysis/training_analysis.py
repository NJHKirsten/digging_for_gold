from analysis.analysis import Analysis
import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class TrainingAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.model_name}_{self.dataset_name}/"
        seeds = json.load(open(f"{path}/seeds.json"))

        data_file_prefix = f"runs/{self.run_config['run']}/training_graphs/{self.model_name}_{self.dataset_name}/"

        metrics = ['testing_accuracy', 'testing_loss', 'training_loss', 'training_accuracy']

        samples = self.__get_samples(data_file_prefix, 'original', self.sample_size, seeds)
        axes = {}
        for metric in metrics:
            ax = samples.plot(use_index=True,
                              y=metric,
                              kind='line',
                              label='original',
                              yerr=f'{metric}_std')
            self.__configure_plot(ax, metric)
            axes[metric] = ax

        for portion in self.run_config["prune_sizes"]:
            samples = self.__get_samples(data_file_prefix, f'{round(portion * 100)}%', self.sample_size, seeds)

            for metric in metrics:
                samples.plot(ax=axes[metric], use_index=True,
                             y=metric,
                             kind='line',
                             label=f'{round(portion * 100)}%',
                             yerr=f'{metric}_std')

        plt.show()

    @staticmethod
    def __get_samples(data_file_prefix, portion_string, sample_size, seeds):
        samples = []
        for sample in range(sample_size):
            samples.append(pd.read_csv(
                f"{data_file_prefix}{seeds[sample]}/{portion_string}.csv"))

        all_data = pd.concat(samples).groupby("epoch")
        deviations = all_data.std().rename(columns={"testing_accuracy": "testing_accuracy_std",
                                                    "testing_loss": "testing_loss_std",
                                                    "training_loss": "training_loss_std",
                                                    "training_accuracy": "training_accuracy_std"})
        means = all_data.mean()
        processed_data = pd.merge(means, deviations, on="epoch")
        return processed_data

    @staticmethod
    def __configure_plot(ax, metric):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric[metric.find('_') + 1:].capitalize())
        ax.set_title(f'{metric.replace("_", " ").capitalize()} during training')
        if metric.find('accuracy') != -1:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
