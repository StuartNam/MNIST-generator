import torch

import matplotlib.pyplot as plt

from model import *
from config import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionModel(NUM_TIMESTEPS)

model.noise_estimator1.load_state_dict(torch.load(MODEL_FOLDER + MODEL1_FILE, map_location = torch.device(device)))
model.noise_estimator2.load_state_dict(torch.load(MODEL_FOLDER + MODEL1_FILE, map_location = torch.device(device)))
model.noise_estimator3.load_state_dict(torch.load(MODEL_FOLDER + MODEL1_FILE, map_location = torch.device(device)))
model.noise_estimator4.load_state_dict(torch.load(MODEL_FOLDER + MODEL1_FILE, map_location = torch.device(device)))

model.eval()
with torch.no_grad():
    x0, xT_0 = model.sample(1)

    x0 = scale_up(x0).reshape(28, 28)
    xT_0 = [scale_up(xt).reshape(28, 28) for xt in xT_0]

    imshow_imgs(xT_0)

    plt.imshow(x0, cmap = 'gray')
    plt.show()

