from torchvision import datasets, transforms
import torch.utils.data
from dataset_setup import DatasetSetup


class Cifar10Setup(DatasetSetup):

    def create_datasets(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        training_set = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        testing_set = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

        return training_set, testing_set
