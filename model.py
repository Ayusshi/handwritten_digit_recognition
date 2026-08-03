"""
model.py

Contains the CNN architecture for handwritten digit recognition.
"""

import torch.nn as nn

from config import (
    INPUT_CHANNELS,
    NUM_CLASSES
)


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    """

    def __init__(self):
        super().__init__()

        # ==========================
        # Feature Extraction
        # ==========================

        self.conv1 = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ==========================
        # Classifier
        # ==========================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 7 * 7,
            128
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            NUM_CLASSES
        )

    def forward(self, x):
        """
        Defines how data flows through the network.
        """

        # Feature Extraction

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten

        x = self.flatten(x)

        # Classification

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x