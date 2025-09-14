import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils


dataset = LightlyDataset("\Users\Besitzer\Desktop\Image_Dataset", transform=None)

dataloader = torch.utils.DataLoader(
    dataset, batch_size=256, shuffle=True,
    drop_last=True, num_workers=8
)

