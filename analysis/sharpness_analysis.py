import copy
import json
import os
import sys
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import torch
from pyhessian import hessian

from analysis.analysis import Analysis
from model_imports import *
from dataset_imports import *
from torch.utils.data.dataloader import default_collate


class SharpnessAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))
        model = self.__class_from_string(self.run_config['model_class'])()

        print(f'Sharpness')
        if self.analysis_config['sharpness_analysis']['hessian']:
            self.__calculate_hessian_sharpness(model, seeds, self.run_config['sample_size'])

        if self.analysis_config['sharpness_analysis']['sampling']:
            for sharpness_config in self.analysis_config['sharpness_analysis']['configs']:
                print(f"Sharpness Config: {sharpness_config['name']}")
                process_list = []
                for sample in range(self.run_config['sample_size']):
                    seed = seeds[sample]
                    print(f"Seed {seed}")
                    if self.analysis_config["multiprocessing"]:
                        process = Process(target=self.calculate_sharpness,
                                          args=(model, seed, sharpness_config))
                        process.start()
                        process_list.append(process)
                        if self.analysis_config["num_processes"] <= len(process_list):
                            for process in process_list:
                                process.join()
                            process_list = []
                    else:
                        self.calculate_sharpness(model, seed, sharpness_config)

                if self.analysis_config["multiprocessing"]:
                    for process in process_list:
                        process.join()

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def calculate_sharpness(self, model, seed, sharpness_config):
        distance = sharpness_config['distance']
        steps = sharpness_config['steps']
        samples = sharpness_config['samples']

        torch.manual_seed(seed)

        model.load_state_dict(
            torch.load(
                f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seed}/original.pt",
                map_location=self.run_config["cuda_device"]),
            strict=False)

        train_loader, test_loader, loss_function, device = self.__inference_setup(model)

        model = model.to(device)
        # train_loss, test_loss = self.__calculate_loss(model,
        #                                               train_loader,
        #                                               test_loader,
        #                                               loss_function,
        #                                               device)
        #
        # print(f"Original Loss: {train_loss}")

        masks = []
        for sample in range(samples):
            masks.append({})
            for name, parameter in model.state_dict().items():
                random_negatives = (torch.rand_like(parameter) < 0.5)
                boolean_mask = torch.rand_like(parameter) < 0.5  # TODO Is it ok if the split is not exactly 50%
                mask = torch.zeros_like(parameter)
                mask[boolean_mask] = 1
                mask[random_negatives] = mask[random_negatives] * -1

                # mask.to(device)
                masks[sample][name] = mask.to(device)

        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/{sharpness_config['name']}/{seed}.csv"
        sharpness_graph = []
        if os.path.isfile(csv_graph_path):
            sharpness_graph = pd.read_csv(csv_graph_path).to_dict('records')

        model_copy = copy.deepcopy(model)

        for sample in range(samples):

            if any([sample == graph['sample'] for graph in sharpness_graph]):
                continue

            # model_copy.load_state_dict(model.state_dict())
            # model_copy.to(device)

            # for name, parameter in model_copy.state_dict().items():
            #     mask = masks[sample][name]  # .to(device)
            #     walk = parameter - (mask * round(distance * steps, 3))
            #     parameter.copy_(walk)

            print(f'[{sample}]')
            for step in range(-1 * steps, steps + 1):
                model_copy.load_state_dict(model.state_dict().copy())
                model_copy.to(device)
                for name, parameter in model_copy.state_dict().items():
                    mask = masks[sample][name]  # .to(device)
                    walk = parameter + (mask * round(distance * step, 3))
                    parameter.copy_(walk)

                train_loss, test_loss = self.__calculate_loss(model_copy,
                                                              train_loader,
                                                              test_loader,
                                                              loss_function,
                                                              device)
                sharpness_graph.append({
                    'sample': sample,
                    'step': round(step * distance, 3),
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
            raise Exception("No CUDA device available")
            # device = torch.device("cpu")

        model.to(device)

        dataset_setup = self.__class_from_string(self.run_config["dataset_class"])()
        training_set, testing_set = dataset_setup.create_datasets()

        train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs,
                                                   collate_fn=lambda x: tuple(
                                                       x_.to(device) for x_ in default_collate(x)))
        test_loader = torch.utils.data.DataLoader(testing_set, **train_kwargs,
                                                  collate_fn=lambda x: tuple(
                                                      x_.to(device) for x_ in default_collate(x)))

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
                output = model(data).to(device)
                train_loss += loss_function(output, target, reduction='sum').item()

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).to(device)
                test_loss += loss_function(output, target, reduction='sum').item()

        train_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return train_loss, test_loss

    def __calculate_hessian_sharpness(self, model, seeds, sample_size):
        csv_graph_path = f"sharpness_results/{self.analysis_config['run']}/hessian.csv"
        sharpness_graph = []
        if os.path.isfile(csv_graph_path):
            sharpness_graph = pd.read_csv(csv_graph_path).to_dict('records')

        concurrent_output_queue = Queue()
        process_list = []
        for sample in range(sample_size):
            seed = seeds[sample]
            if any([seed == graph['seed'] for graph in sharpness_graph]):
                continue

            process = Process(target=self.calculate_sample_hessian,
                              args=(model, seed, concurrent_output_queue))
            process.start()
            process_list.append(process)
            if self.analysis_config["num_processes"] <= len(process_list):
                for process in process_list:
                    process.join()
                process_list = []
                while not concurrent_output_queue.empty():
                    sharpness_graph.append(concurrent_output_queue.get_nowait())
            # self.calculate_sample_hessian(model, seed, concurrent_output_queue)
            if sharpness_graph:
                os.makedirs(os.path.dirname(csv_graph_path), exist_ok=True)
                pd.DataFrame(sharpness_graph).to_csv(csv_graph_path, index=False)

        if self.analysis_config["multiprocessing"]:
            for process in process_list:
                process.join()
            while not concurrent_output_queue.empty():
                sharpness_graph.append(concurrent_output_queue.get_nowait())
        if sharpness_graph:
            os.makedirs(os.path.dirname(csv_graph_path), exist_ok=True)
            pd.DataFrame(sharpness_graph).to_csv(csv_graph_path, index=False)

    def calculate_sample_hessian(self, model, seed, sharpness_graph):

        model = copy.deepcopy(model)
        model.load_state_dict(
            torch.load(
                f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seed}/original.pt",
                map_location=self.run_config["cuda_device"]),
            strict=False)
        model.eval()
        train_loader, test_loader, loss_function, device = self.__inference_setup(model)
        model = model.to(device)

        torch.manual_seed(seed)

        print("start")
        hessian_comp = hessian(model, loss_function, dataloader=train_loader, cuda=True)
        print("hessian")
        hessian_trace = np.mean(hessian_comp.trace(tol=1e-3))
        print("trace")
        hessian_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1, tol=1e-3)
        hessian_top_eigenvalue = hessian_eigenvalues[0]
        print(f"[{seed}] - {hessian_trace} - {hessian_top_eigenvalue}")
        sharpness_graph.put({
            'seed': seed,
            'hessian_trace': hessian_trace,
            'hessian_top_eigenvalue': hessian_top_eigenvalue
        })
