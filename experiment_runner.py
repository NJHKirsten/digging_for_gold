import os
import sys

import pandas as pd
import json
import torch
import torch.nn.utils.prune as prune

from model_imports import *
from dataset_imports import *


class ExperimentRunner:
    experiment_configs = None

    def __init__(self, experiment_configs, global_config):
        self.experiment_configs = experiment_configs
        self.global_config = global_config

    @staticmethod
    def run():
        experiment_configs = pd.read_csv("experiment_config.csv")
        global_config = json.load(open("all_experiments_config.json"))
        global_config['run'] = ExperimentRunner.__get_run_number()
        experiment_runner = ExperimentRunner(experiment_configs, global_config)
        experiment_runner.run_all()

    @staticmethod
    def __get_run_number():
        if not os.path.exists('runs'):
            os.makedirs('runs')
            return 0
        existing_runs = os.listdir('runs')
        existing_runs.sort(reverse=True)
        return 0 if len(existing_runs) == 0 else int(existing_runs[0]) + 1

    def run_all(self):
        for model_name, model_class, dataset_name, dataset_class, sample_size in zip(
                self.experiment_configs["model_name"],
                self.experiment_configs["model_class"],
                self.experiment_configs["dataset_name"],
                self.experiment_configs["dataset_class"],
                self.experiment_configs["sample_size"]):
            print(f"Running experiment for {model_name} on {dataset_name}")
            self.__run_training_experiment(model_name, model_class, dataset_name, dataset_class, sample_size)
            self.__run_pruning_experiment(model_name, model_class, dataset_name, dataset_class, sample_size)

    def __run_training_experiment(self, model_name, model_class, dataset_name, dataset_class, sample_size):
        dataset_setup = self.__class_from_string(dataset_class)()

        path = f"experiments/{model_name}_{dataset_name}/"
        parameters = json.load(open(f"{path}/parameters.json"))
        seeds = json.load(open(f"{path}/seeds.json"))

        for sample in range(sample_size):
            seed = seeds[sample]
            print(f"Running sample {sample} with seed {seed}")
            model = self.__class_from_string(model_class)()
            self.__train_sample(model, model_name, parameters, dataset_setup, dataset_name, seed)

    def __run_pruning_experiment(self, model_name, model_class, dataset_name, dataset_class, sample_size):
        dataset_setup = self.__class_from_string(dataset_class)()

        path = f"experiments/{model_name}_{dataset_name}/"
        parameters = json.load(open(f"{path}/parameters.json"))
        seeds = json.load(open(f"{path}/seeds.json"))

        for sample in range(sample_size):
            seed = seeds[sample]
            print(f"Pruning sample {sample} with seed {seed}")
            model = self.__class_from_string(model_class)()
            self.__prune_sample(model, model_name, parameters, dataset_setup, dataset_name, seed)

    def __prune_sample(self, model, model_name, parameters, dataset_setup, dataset_name, seed):

        model.load_state_dict(torch.load(f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/original.pt"))
        torch.manual_seed(seed)
        parameters_to_prune = model.get_pruning_parameters()

        for portion in self.global_config["prune_sizes"]:

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=portion,
            )
            for module in parameters_to_prune:
                prune.remove(module, 'weight')

            print(f"Pruned {portion * 100}% of weights")
            self.__train_sample(model,
                                model_name,
                                parameters,
                                dataset_setup,
                                dataset_name,
                                seed,
                                f'{round(portion * 100)}%')
            model.load_state_dict(
                torch.load(f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/{round(portion * 100)}%.pt"))
            torch.manual_seed(seed)

    def __train_sample(self, model, model_name, parameters, dataset_setup, dataset_name, seed, portion="original"):

        torch.manual_seed(seed)

        train_kwargs = {'batch_size': parameters["batch_size"]}
        test_kwargs = {'batch_size': parameters["test_batch_size"]}

        if parameters["use_cuda"] and torch.cuda.is_available():
            device = torch.device(parameters["cuda_device"])
            cuda_kwargs = parameters["cuda_config"]
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        else:
            device = torch.device("cpu")

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

        os.makedirs(f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}", exist_ok=True)
        torch.save(model.state_dict(),
                   f"runs/{self.global_config['run']}/trained_models/{model_name}_{dataset_name}/{seed}/{portion}.pt")
        training_graph_df = pd.DataFrame(training_graph)
        os.makedirs(f"runs/{self.global_config['run']}/training_graphs/{model_name}_{dataset_name}/{seed}", exist_ok=True)
        training_graph_df.to_csv(f"runs/{self.global_config['run']}/training_graphs/{model_name}_{dataset_name}/{seed}/{portion}.csv", index=False)

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
