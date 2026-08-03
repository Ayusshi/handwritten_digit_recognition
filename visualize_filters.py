"""
visualize_filters.py

Visualize the learned filters of the first convolution layer.
"""

import torch
import matplotlib.pyplot as plt

from config import DEVICE, MODEL_PATH
from model import CNN


def normalize_filter(filter_tensor):
    """
    Normalize filter values to [0,1] for visualization.
    """

    filter_tensor = filter_tensor - filter_tensor.min()
    filter_tensor = filter_tensor / filter_tensor.max()

    return filter_tensor


def main():

    # ==========================
    # Load Model
    # ==========================

    model = CNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    # ==========================
    # Get Conv1 Filters
    # Shape:
    # [32,1,3,3]
    # ==========================

    filters = model.conv1.weight.detach().cpu()

    # ==========================
    # Display Filters
    # ==========================

    fig, axes = plt.subplots(
        4,
        8,
        figsize=(12,6)
    )

    for i, ax in enumerate(axes.flat):

        filter_img = filters[i, 0]

        filter_img = normalize_filter(filter_img)

        ax.imshow(
            filter_img,
            cmap="gray"
        )

        ax.set_title(f"F{i}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()