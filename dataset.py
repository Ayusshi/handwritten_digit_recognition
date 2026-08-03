"""
dataset.py

Creates the MNIST datasets and DataLoaders.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    DATA_DIR,
    BATCH_SIZE
)


# ======================================
# Training Transform
# ======================================

train_transform = transforms.Compose([
    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ToTensor()
])


# ======================================
# Testing Transform
# ======================================

test_transform = transforms.Compose([
    transforms.ToTensor()
])


def get_data_loaders():
    """
    Creates and returns the training and testing DataLoaders.

    Returns:
        train_loader (DataLoader)
        test_loader (DataLoader)
    """

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader