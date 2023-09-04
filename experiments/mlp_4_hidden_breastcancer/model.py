import torch
import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Mlp4HiddenBreastCancer(nn.Module):
    def __init__(self):
        super(Mlp4HiddenBreastCancer, self).__init__()
        self.fc1 = nn.Linear(30, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.fc2(x)
        x = functional.relu(x)
        x = self.fc3(x)
        x = functional.relu(x)
        x = self.fc4(x)
        x = functional.relu(x)
        x = self.fc5(x)
        output = functional.sigmoid(x)
        return output

    def get_training_config(self, learning_rate, gamma, optimizer="adam"):
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        loss_function = functional.binary_cross_entropy
        return dict(
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function
        )

    def get_pruning_parameters(self):
        return (
            (self.fc1, 'weight'),
            (self.fc2, 'weight'),
            (self.fc3, 'weight'),
            (self.fc4, 'weight'),
            (self.fc5, 'weight'),
            (self.fc1, 'bias'),
            (self.fc2, 'bias'),
            (self.fc3, 'bias'),
            (self.fc4, 'bias'),
            (self.fc5, 'bias'),
        )
