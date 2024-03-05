import numpy as np

from config_loader import load_config
from trainer import train

cfg = load_config()

# example image
img = np.load(f'{cfg.data_dir}/small_brain.npy')

train(img, cfg)