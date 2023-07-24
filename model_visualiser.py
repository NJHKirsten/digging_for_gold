import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.mnist import MnistSetup
from experiments.mlp_2_hidden_mnist.model import Mlp2Hidden


class ModelVisualiser:

    @staticmethod
    def visualise():
        model = Mlp2Hidden()
        training_set, testing_set = MnistSetup().create_datasets()
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=False)
        images, labels = next(iter(train_loader))

        writer = SummaryWriter('runs/mlp_2_hidden_mnist')
        writer.add_graph(model, images)
        writer.close()
