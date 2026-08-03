"""
train.py

Trains the CNN model on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    MODEL_SAVE_PATH
)

from dataset import get_data_loaders
from model import CNN

from utils import (
    get_device,
    save_model,
    count_parameters,
    calculate_accuracy
)


def train():

    # ==========================================
    # Device
    # ==========================================

    device = get_device()

    print(f"Using device: {device}")

    # ==========================================
    # Data
    # ==========================================

    train_loader, test_loader = get_data_loaders()

    # ==========================================
    # Model
    # ==========================================

    model = CNN().to(device)

    print(model)

    print(f"\nTrainable Parameters: {count_parameters(model):,}")

    # ==========================================
    # Loss Function
    # ==========================================

    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # Optimizer
    # ==========================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==========================================
    # Training
    # ==========================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ======================================
        # Evaluation
        # ======================================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                correct += calculate_accuracy(
                    outputs,
                    labels
                )

                total += labels.size(0)

        accuracy = (correct / total) * 100

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    # ==========================================
    # Save Model
    # ==========================================

    save_model(
        model,
        MODEL_SAVE_PATH
    )

    print("\nTraining Finished Successfully!")


if __name__ == "__main__":
    train()