import json
import sys

import torch
from torch import nn

from analysis.analysis import Analysis
from model_imports import *


class WeightAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.model_name}_{self.dataset_name}/"
        seeds = json.load(open(f"{path}/seeds.json"))
        model = self.__class_from_string(self.model_class)()

        for portion in ['original']+self.global_config["prune_sizes"]:
            portion_name = portion if portion == 'original' else f'{round(portion * 100)}%'

            print(f'portion_name zero weights')
            for sample in range(self.sample_size):
                model.load_state_dict(
                    torch.load(
                        f"runs/{self.global_config['run']}/trained_models/{self.model_name}_{self.dataset_name}/{seeds[sample]}/{portion_name}.pt"),
                    strict=False)
                zero_weights, total_weights = self.__get_zero_weights(model)
                print(f"[{sample}] - {zero_weights}/{total_weights} - {zero_weights / total_weights}")

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    @staticmethod
    def __get_zero_weights(model):
        zero_weights = 0
        total_weights = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                zero_weights += torch.sum(param.abs() < 0.0001)
                total_weights += param.numel()
        return zero_weights, total_weights
