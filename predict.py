"""
predict.py

Loads a trained CNN model and predicts
a single handwritten digit image.
"""

import torch
from PIL import Image
from torchvision import transforms

from model import CNN

from config import (
    MODEL_SAVE_PATH,
    IMAGE_SIZE
)

from utils import (
    get_device,
    load_model
)


def predict(image_path):
    """
    Predicts a handwritten digit from an image.

    Args:
        image_path (str): Path to image.
    """

    # ==========================================
    # Device
    # ==========================================

    device = get_device()

    # ==========================================
    # Load Model
    # ==========================================

    model = CNN()

    model = load_model(
        model,
        MODEL_SAVE_PATH,
        device
    )

    # ==========================================
    # Image Transform
    # ==========================================

    transform = transforms.Compose([
        transforms.Grayscale(),

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor()
    ])

    # ==========================================
    # Load Image
    # ==========================================

    image = Image.open(image_path)

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)

    # ==========================================
    # Prediction
    # ==========================================

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    print(f"\nPredicted Digit : {prediction.item()}")

    print(
        f"Confidence      : {confidence.item()*100:.2f}%"
    )


if __name__ == "__main__":

    IMAGE_PATH = "test_images/digit.png"

    predict(IMAGE_PATH)