"""
config.py

Contains all project configuration variables.

Changing values here automatically affects the entire project.
"""

# ==========================
# Dataset
# ==========================

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
# Image Properties
# ==========================

IMAGE_SIZE = 28

# ==========================
# Paths
# ==========================

MODEL_SAVE_PATH = "saved_models/mnist_cnn.pth"

# ==========================
# Random Seed
# ==========================

RANDOM_SEED = 42