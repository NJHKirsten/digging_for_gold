import math
import os
import sys

import pandas as pd
import json
import torch
import torch.nn.utils.prune as prune
from torch.nn import init

from analysis.weight_analysis import WeightAnalysis
from model_imports import *
from dataset_imports import *

from multiprocessing import Process


class ExperimentRunner:
    experiment_configs = None

    def __init__(self, experiment_configs, global_config, seeds_file="seeds.json"):
        self.experiment_configs = experiment_configs
        self.global_config = global_config
        self.seeds_file = seeds_file

    @staticmethod
    def run(only_pruning=False, experiments_file="experiment_config.csv",
            global_config_file="all_experiments_config.json", seeds_file="seeds.json"):
        experiment_configs = pd.read_csv(f'setups/{experiments_file}')
        global_config = json.load(open(f'config/{global_config_file}'))
        if not only_pruning:
            global_config['run'] = ExperimentRunner.__get_run_number() + 1
        else:
            global_config['run'] = ExperimentRunner.__get_run_number()

        experiment_runner = ExperimentRunner(experiment_configs, global_config, seeds_file)
        experiment_runner.run_all(only_pruning)

    @staticmethod
    def __get_run_number():
        if not os.path.exists('runs'):
            os.makedirs('runs')
            return 0
        existing_runs = os.listdir('runs')
        existing_runs.sort(reverse=True)
        return 0 if len(existing_runs) == 0 else int(existing_runs[0])

    def run_all(self, only_pruning=False):
        for model_name, model_class, dataset_name, dataset_class, sample_size, parameters in zip(
                self.experiment_configs["model_name"],
                self.experiment_configs["model_class"],
                self.experiment_configs["dataset_name"],
                self.experiment_configs["dataset_class"],
                self.experiment_configs["sample_size"],
                self.experiment_configs["parameters"]):
            print(f"Running experiment for {model_name} on {dataset_name}")
            if not only_pruning:
                self.__run_training_experiment(model_name, model_class, dataset_name, dataset_class, sample_size,
                                               parameters)
            self.__run_pruning_experiment(model_name, model_class, dataset_name, dataset_class, sample_size, parameters)

    def __run_training_experiment(self, model_name, model_class, dataset_name, dataset_class, sample_size, parameters):
        dataset_setup = self.__class_from_string(dataset_class)()

        path = f"experiments/{model_name}_{dataset_name}/"
        parameters = json.load(open(f"{path}/{parameters}"))
        seeds = json.load(open(f"{path}/{self.seeds_file}"))

        process_list = []
        for sample in range(sample_size):
            seed = seeds[sample]
            print(f"Running sample {sample} with seed {seed}")
            model = self.__class_from_string(model_class)()
            if self.global_config["multiprocessing"]:
                process = Process(target=self.train_sample,
                                  args=(model, model_name, parameters, dataset_setup, dataset_name, seed))
                process.start()
                process_list.append(process)
            else:
                self.train_sample(model, model_name, parameters, dataset_setup, dataset_name, seed)

        if self.global_config["multiprocessing"]:
            for process in process_list:
                process.join()

    def __run_pruning_experiment(self, model_name, model_class, dataset_name, dataset_class, sample_size, parameters):
        dataset_setup = self.__class_from_string(dataset_class)()

        path = f"experiments/{model_name}_{dataset_name}/"
        parameters = json.load(open(f"{path}/{parameters}"))
        seeds = json.load(open(f"{path}/{self.seeds_file}"))

        process_list = []
        for sample in range(sample_size):
            seed = seeds[sample]
            print(f"Pruning sample {sample} with seed {seed}")
            model = self.__class_from_string(model_class)()

            if self.global_config["multiprocessing"]:
                process = Process(target=self.prune_sample,
                                  args=(model, model_name, parameters, dataset_setup, dataset_name, seed))
                process.start()
                process_list.append(process)
            else:
                self.prune_sample(model, model_name, parameters, dataset_setup, dataset_name, seed)

        if self.global_config["multiprocessing"]:
            for process in process_list:
                process.join()

    def prune_sample(self, model, model_name, parameters, dataset_setup, dataset_name, seed):

        model.load_state_dict(torch.load(
            f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/original.pt"))
        torch.manual_seed(seed)
        parameters_to_prune = model.get_pruning_parameters()

        for portion in self.global_config["prune_sizes"]:

            # TODO DELETE
            zero_weights, total_weights = WeightAnalysis.get_zero_weights(model)
            print(f"Before Pruning - {zero_weights}/{total_weights} - {zero_weights / total_weights}")
            ############################

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - portion,
            )
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         param.data.random_()
            for layer in model.children():
                self.__reinitialise_pruned_layer(layer)
            # for module, name in parameters_to_prune:
            #     prune.remove(module, name)

            print(f"Pruned {portion * 100}% of weights")

            # TODO DELETE
            zero_weights, total_weights = WeightAnalysis.get_zero_weights(model)
            print(f"After pruning - {zero_weights}/{total_weights} - {zero_weights / total_weights}")
            ############################

            self.train_sample(model,
                              model_name,
                              parameters,
                              dataset_setup,
                              dataset_name,
                              seed,
                              f'{round(portion * 100)}%')

            # TODO DELETE
            zero_weights, total_weights = WeightAnalysis.get_zero_weights(model)
            print(f"Saving model - {zero_weights}/{total_weights} - {zero_weights / total_weights}")
            ############################

            model.load_state_dict(
                torch.load(
                    f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/{round(portion * 100)}%.pt"))
            torch.manual_seed(seed)

    @staticmethod
    def __reinitialise_pruned_layer(layer):
        if prune.is_pruned(layer):
            init.kaiming_uniform_(layer.weight_orig, a=math.sqrt(5))
            prune.custom_from_mask(layer, 'weight', layer.weight_mask)
            if layer.bias_orig is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight_orig)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(layer.bias_orig, -bound, bound)
                prune.custom_from_mask(layer, 'bias', layer.bias_mask)
        else:
            layer.reset_parameters()

    def train_sample(self, model, model_name, parameters, dataset_setup, dataset_name, seed, portion="original"):

        torch.manual_seed(seed)

        train_kwargs = {'batch_size': parameters["batch_size"]}
        test_kwargs = {'batch_size': parameters["test_batch_size"]}

        if parameters["use_cuda"] and torch.cuda.is_available():
            device = torch.device(parameters["cuda_device"])
            cuda_kwargs = parameters["cuda_config"]
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        else:
            raise Exception("No CUDA device available")
            # device = torch.device("cpu")

        training_set, testing_set = dataset_setup.create_datasets()

        train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(testing_set, **test_kwargs)

        model = model.to(device)

        model_training_config = model.get_training_config(parameters["learning_rate"], parameters["gamma"])
        optimizer = model_training_config["optimizer"]
        scheduler = model_training_config["scheduler"]
        loss_function = model_training_config["loss_function"]

        training_graph = []

        for epoch in range(1, parameters["epochs"] + 1):
            self.__train_model(model, device, train_loader, optimizer, epoch, loss_function, parameters["log_interval"],
                               parameters["dry_run"])
            self.__test_model(model, device, test_loader, loss_function)
            training_graph.append(self.__get_epoch_data(epoch, model, device, train_loader, test_loader, loss_function))
            scheduler.step()
            if epoch % parameters["log_interval"] == 0:
                print(f"Finished epoch {epoch}")

        os.makedirs(f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}",
                    exist_ok=True)

        # TODO DELETE
        zero_weights, total_weights = WeightAnalysis.get_zero_weights(model)
        print(f"After training - {zero_weights}/{total_weights} - {zero_weights / total_weights}")
        ############################
        torch.save(model.state_dict(),
                   f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/{portion}.pt")
        training_graph_df = pd.DataFrame(training_graph)
        os.makedirs(f"runs/{self.global_config['run']}/training_graphs/{model_name}_{dataset_name}/{seed}",
                    exist_ok=True)
        training_graph_df.to_csv(
            f"runs/{self.global_config['run']}/training_graphs/{model_name}_{dataset_name}/{seed}/{portion}.csv",
            index=False)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    @staticmethod
    def __train_model(model, device, train_loader, optimizer, epoch, loss_function, log_interval, dry_run):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.item()))
                total_loss += loss.item()
                if dry_run:
                    break

    @staticmethod
    def __test_model(model, device, test_loader, loss_function):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_function(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))

    @staticmethod
    def __get_epoch_data(epoch, model, device, train_loader, test_loader, loss_function):
        training_loss = 0
        train_correct = 0
        testing_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                training_loss += loss_function(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                testing_loss += loss_function(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        training_accuracy = 100. * train_correct / len(train_loader.dataset)
        testing_accuracy = 100. * test_correct / len(test_loader.dataset)
        return {"epoch": epoch,
                "training_loss": training_loss,
                "testing_loss": testing_loss,
                "training_accuracy": training_accuracy,
                "testing_accuracy": testing_accuracy}
