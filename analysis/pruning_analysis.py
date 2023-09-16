import json
import sys

import torch

from analysis.analysis import Analysis
from model_imports import *


class PruningAnalysis(Analysis):

    def run(self):
        print("Pruning Analysis")
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))
        model = self.__class_from_string(self.run_config['model_class'])()

        for portion in ['original'] + self.run_config["prune_sizes"]:
            portion_name = portion if portion == 'original' else f'{round(portion * 100)}%'

            print(f'{portion_name} zero weights')
            for sample in range(self.run_config['sample_size']):
                model.load_state_dict(
                    torch.load(
                        f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seeds[sample]}/{portion_name}.pt",
                        map_location=self.run_config["cuda_device"]),
                    strict=False)
                zero_weights, total_weights = self.get_zero_weights(model)
                print(f"[{sample}] - {zero_weights}/{total_weights} - {zero_weights / total_weights}")

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    @staticmethod
    def get_zero_weights(model):
        zero_weights = 0
        total_weights = 0
        for layer in model.children():
            zero_weights += torch.sum(layer.weight.abs() < 0.000001)
            zero_weights += torch.sum(layer.bias.abs() < 0.000001)
            total_weights += layer.weight.numel()
            total_weights += layer.bias.numel()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         zero_weights += torch.sum(param.abs() < 0.000001)
        #         total_weights += param.numel()
        return zero_weights, total_weights
