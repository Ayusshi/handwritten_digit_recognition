"""
visualize_features.py

Visualize feature maps learned by the first convolution layer.
"""

import torch
import matplotlib.pyplot as plt

from config import DEVICE, MODEL_PATH
from model import CNN
from preprocess import preprocess_image
from PIL import Image
import torchvision.transforms as transforms

# Dictionary to store activations
activations = {}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
def save_activation(name):
    """
    Creates a forward hook that stores layer outputs.
    """

    def hook(module, input, output):
        activations[name] = output.detach().cpu()

    return hook


def main():

    # ==========================
    # Load Model
    # ==========================

    model = CNN().to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    # ==========================
    # Load Image
    # ==========================

    processed = preprocess_image("test_images/digit1.jpg")

    image = Image.fromarray(processed)

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    # ==========================
    # Register Hook
    # ==========================

    handle = model.conv2.register_forward_hook(
        save_activation("conv2")
    )

    # ==========================
    # Run Forward Pass
    # ==========================

    with torch.no_grad():
        _ = model(image)

    # ==========================
    # Remove Hook
    # ==========================

    handle.remove()

    # ==========================
    # Extract Feature Maps
    # Shape:
    # [32, 28, 28]
    # ==========================

    feature_maps = activations["conv2"][0]

    # ==========================
    # Display Feature Maps
    # ==========================

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(8, 8)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            feature_maps[i],
            cmap="gray"
        )

        ax.set_title(f"Filter {i}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()