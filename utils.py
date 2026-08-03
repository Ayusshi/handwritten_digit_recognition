"""
utils.py

Utility/helper functions used throughout the project.
"""

import torch


def get_device():
    """
    Returns the available device.

    Returns:
        torch.device
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

import os

def save_model(model, model_path):

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(f"\n✅ Model saved to: {model_path}")


def load_model(model, model_path, device):
    """
    Loads saved model weights.

    Args:
        model: Model architecture
        model_path: Saved model path
        device: CPU or GPU

    Returns:
        Loaded model
    """

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model


def count_parameters(model):
    """
    Returns total trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        int
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_accuracy(outputs, labels):
    """
    Calculates batch accuracy.

    Args:
        outputs: Model predictions (logits)
        labels: Ground truth labels

    Returns:
        Number of correct predictions
    """

    _, predicted = torch.max(outputs, dim=1)

    correct = (predicted == labels).sum().item()

    return correct