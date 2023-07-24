import torch
import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = functional.log_softmax(x, dim=1)
        return output

    def get_training_config(self, learning_rate, gamma):
        optimizer = optim.Adadelta(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        loss_function = functional.nll_loss
        return dict(
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function
        )

    def get_pruning_parameters(self):
        return (
            (self.fc1, 'weight'),
            (self.fc2, 'weight'),
        )
