import torch
import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Mlp10Hidden(nn.Module):
    def __init__(self):
        super(Mlp10Hidden, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 256)
        self.fc9 = nn.Linear(256, 128)
        self.fc10 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.fc2(x)
        x = functional.relu(x)
        x = self.fc3(x)
        x = functional.relu(x)
        x = self.fc4(x)
        x = functional.relu(x)
        x = self.fc5(x)
        x = functional.relu(x)
        x = self.fc6(x)
        x = functional.relu(x)
        x = self.fc7(x)
        x = functional.relu(x)
        x = self.fc8(x)
        x = functional.relu(x)
        x = self.fc9(x)
        x = functional.relu(x)
        x = self.fc10(x)
        output = functional.log_softmax(x, dim=1)
        return output

    def get_training_config(self, learning_rate, gamma, optimizer="adam"):
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adadelta':
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
            (self.fc3, 'weight'),
            (self.fc4, 'weight'),
            (self.fc5, 'weight'),
            (self.fc6, 'weight'),
            (self.fc7, 'weight'),
            (self.fc8, 'weight'),
            (self.fc9, 'weight'),
            (self.fc10, 'weight'),
            (self.fc1, 'bias'),
            (self.fc2, 'bias'),
            (self.fc3, 'bias'),
            (self.fc4, 'bias'),
            (self.fc5, 'bias'),
            (self.fc6, 'bias'),
            (self.fc7, 'bias'),
            (self.fc8, 'bias'),
            (self.fc9, 'bias'),
            (self.fc10, 'bias'),
        )
