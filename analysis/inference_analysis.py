import copy
import json
import sys

import torch

from analysis.analysis import Analysis
from model_imports import *
from dataset_imports import *


class InferenceAnalysis(Analysis):

    def run(self):
        path = f"experiments/{self.run_config['model_name']}_{self.run_config['dataset_name']}/"
        seeds = json.load(open(f"{path}/{self.seeds_file}"))
        model = self.__class_from_string(self.run_config['model_class'])()

        print(f'Inference accuracy')
        for sample in range(self.run_config['sample_size']):
            seed = seeds[sample]
            model.load_state_dict(
                torch.load(
                    f"runs/{self.analysis_config['run']}/trained_models/{self.run_config['model_name']}_{self.run_config['dataset_name']}/{seed}/original.pt"),
                strict=False)
            print(f"Seed {seed}")
            self.__calculate_inference_accuracy(model, seed)

    @staticmethod
    def __class_from_string(class_name):
        return getattr(sys.modules[__name__], class_name)

    def __calculate_inference_accuracy(self, model, seed):

        train_loader, test_loader, device, loss_function = self.__inference_setup(model)

        training_accuracy, testing_accuracy, training_loss, testing_loss = self.__calculate_accuracy(model, train_loader, test_loader, device, loss_function)

        print(f"Training accuracy: {training_accuracy*100:.2f}%")
        print(f"Testing accuracy: {testing_accuracy*100:.2f}%")
        print(f"Training loss: {training_loss:.2f}")
        print(f"Testing loss: {testing_loss:.2f}")

    def __inference_setup(self, model):
        train_kwargs = {'batch_size': self.run_config["batch_size"]}

        if self.run_config["use_cuda"] and torch.cuda.is_available():
            device = torch.device(self.run_config["cuda_device"])
            cuda_kwargs = self.run_config["cuda_config"]
            train_kwargs.update(cuda_kwargs)
        else:
            raise Exception("No CUDA device available")
            # device = torch.device("cpu")

        dataset_setup = self.__class_from_string(self.run_config["dataset_class"])()
        training_set, testing_set = dataset_setup.create_datasets()

        train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(testing_set, **train_kwargs)

        model_training_config = model.get_training_config(self.run_config["learning_rate"], self.run_config["gamma"],
                                                          self.run_config["optimizer"])
        loss_function = model_training_config["loss_function"]

        return train_loader, test_loader, device, loss_function

    def __calculate_accuracy(self, model, train_loader, test_loader, device, loss_function):
        # train_loader, loss_function, device = self.__inference_setup(model)
        model.to(device)
        train_correct = 0
        test_correct = 0
        training_loss = 0
        testing_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).to(device)
                training_loss += loss_function(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).to(device)
                testing_loss += loss_function(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        train_accuracy = train_correct * 1. / len(train_loader.dataset)
        test_accuracy = test_correct * 1. / len(test_loader.dataset)
        return train_accuracy, test_accuracy, training_loss, testing_loss
