from torchvision import datasets, transforms
from dataset_setup import DatasetSetup


class MnistSetup(DatasetSetup):

    def create_datasets(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        training_set = datasets.MNIST('data', train=True, download=True,
                                      transform=transform)
        testing_set = datasets.MNIST('data', train=False,
                                     transform=transform)
        return training_set, testing_set
