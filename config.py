"""
config.py

Contains all project configuration variables.

Changing values here automatically affects the entire project.
"""

# ==========================
# Dataset
# ==========================

import torch


DATA_DIR = "data"

# ==========================
# Training Hyperparameters
# ==========================

BATCH_SIZE = 64

LEARNING_RATE = 0.001

NUM_EPOCHS = 5

# ==========================
# Model
# ==========================

NUM_CLASSES = 10

INPUT_CHANNELS = 1

# ==========================
# Device
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Image Properties
# ==========================

IMAGE_SIZE = 28

# ==========================
# Paths
# ==========================

MODEL_PATH = "saved_models/mnist_cnn.pth"
MODEL_SAVE_PATH = MODEL_PATH

# ==========================
# Random Seed
# ==========================

RANDOM_SEED = 42