from torchvision import datasets, transforms
from dataset_setup import DatasetSetup


class FashionMnistSetup(DatasetSetup):

    def create_datasets(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        training_set = datasets.FashionMNIST('data', train=True, download=True,
                                             transform=transform)
        testing_set = datasets.FashionMNIST('data', train=False,
                                            transform=transform)
        return training_set, testing_set
