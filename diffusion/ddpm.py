import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def extract(a, t, x_shape):