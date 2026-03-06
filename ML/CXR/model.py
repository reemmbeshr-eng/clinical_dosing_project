"""Input Image (128x128)

Conv (3 → 32)
BatchNorm
ReLU
MaxPool

Conv (32 → 64)
BatchNorm
ReLU
MaxPool

Conv (64 → 128)
BatchNorm
ReLU
MaxPool

Flatten

Fully Connected (128)
Dropout

Output (2 classes)"""


import torch
import torch.nn as nn

class PneumoniaCNN(nn.Module):

    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)

        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x