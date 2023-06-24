import torch

# General configuration
MODEL_STATE_DICT_PATH = "model.pt"

# Model configuration
NUM_TIMESTEPS = 100
DTYPE = torch.float32

# Training configuration
NUM_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# General configuration
RESULT_PATH = "result/"