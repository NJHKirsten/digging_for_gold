import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 80, kernel_size=5, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(80, 96, kernel_size=5, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=5, padding=4),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 196, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 196)
        x = self.classifier(x)
        return x

    def get_training_config(self, learning_rate, gamma, optimizer="adam"):
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        loss_function = functional.cross_entropy
        return dict(
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function
        )

    def get_pruning_parameters(self):
        return (
            (self.features, 'weight'),
            (self.classifier, 'weight'),
            (self.features, 'bias'),
            (self.classifier, 'bias'),
        )


