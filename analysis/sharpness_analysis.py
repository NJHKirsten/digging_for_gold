import copy
import json
import os
import sys

import pandas as pd
import torch

from analysis.analysis import Analysis
from model_imports import *
from dataset_imports import *


class SharpnessAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))
        model = self.__class_from_string(self.run_config['model_class'])()

        print(f'Sharpness')
        for sharpness_config in self.analysis_config['sharpness_analysis']['configs']:
            print(f"Sharpness Config: {sharpness_config['name']}")
            for sample in range(self.run_config['sample_size']):
                seed = seeds[sample]
                model.load_state_dict(
                    torch.load(
                        f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seed}/original.pt"),
                    strict=False)
                print(f"Seed {seed}")
                self.__calculate_sharpness(model, seed, sharpness_config)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __calculate_sharpness(self, model, seed, sharpness_config):
        distance = sharpness_config['distance']
        steps = sharpness_config['steps']
        samples = sharpness_config['samples']

        torch.manual_seed(seed)

        model_copy = copy.deepcopy(model)
        train_loader, test_loader, loss_function, device = self.__inference_setup(model)

        masks = []
        for sample in range(samples):
            masks.append({})
            for name, parameter in model_copy.state_dict().items():
                mask = torch.rand_like(parameter) < 0.5  # TODO Is it ok if the split is not exactly 50%
                # mask.to(device)
                masks[sample][name] = mask.to(device)

        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{seed}.csv"
        sharpness_graph = []
        if os.path.isfile(csv_graph_path):
            sharpness_graph = pd.read_csv(csv_graph_path).to_dict('records')

        model_copy.to(device)

        for sample in range(samples):

            if any([sample == graph['sample'] for graph in sharpness_graph]):
                continue

            for name, parameter in model_copy.state_dict().items():
                mask = masks[sample][name]  # .to(device)
                walk = parameter - (mask * distance * steps)
                parameter.copy_(walk)

            print(f'[{sample}]')
            for step in range(-1 * steps, steps + 1):
                # model_copy.load_state_dict(model.state_dict())
                for name, parameter in model_copy.state_dict().items():
                    mask = masks[sample][name]  # .to(device)
                    walk = parameter + (mask * distance)
                    parameter.copy_(walk)

                train_loss, test_loss = self.__calculate_loss(model_copy,
                                                              train_loader,
                                                              test_loader,
                                                              loss_function,
                                                              device)
                sharpness_graph.append({
                    'sample': sample,
                    'step': step * distance,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                })
                print(f"{step * distance:.3g} - {train_loss}")

            os.makedirs(os.path.dirname(csv_graph_path), exist_ok=True)
            pd.DataFrame(sharpness_graph).to_csv(csv_graph_path, index=False)

    def __inference_setup(self, model):
        train_kwargs = {'batch_size': self.run_config["batch_size"]}

        if self.run_config["use_cuda"] and torch.cuda.is_available():
            device = torch.device(self.run_config["cuda_device"])
            cuda_kwargs = self.run_config["cuda_config"]
            train_kwargs.update(cuda_kwargs)
        else:
            # raise Exception("No CUDA device available")
            device = torch.device("cpu")

        dataset_setup = self.__class_from_string(self.run_config["dataset_class"])()
        training_set, testing_set = dataset_setup.create_datasets()

        train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(testing_set, **train_kwargs)

        model_training_config = model.get_training_config(self.run_config["learning_rate"], self.run_config["gamma"],
                                                          self.run_config["optimizer"])
        loss_function = model_training_config["loss_function"]

        return train_loader, test_loader, loss_function, device

    def __calculate_loss(self, model, train_loader, test_loader, loss_function, device):
        # train_loader, loss_function, device = self.__inference_setup(model)
        # model.to(device)
        model.eval()
        train_loss = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += loss_function(output, target, reduction='sum').item()

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_function(output, target, reduction='sum').item()

        return train_loss, test_loss
