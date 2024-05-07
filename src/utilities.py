import os
import torch
import plotly.graph_objects as go
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from openpyxl import Workbook

def minmax_scale(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return scaled_tensor, tensor_min, tensor_max

def inverse_minmax_scale(scaled_tensor, tensor_min, tensor_max):
    original_tensor = scaled_tensor * (tensor_max - tensor_min) + tensor_min
    return original_tensor

class SlidingWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size=24, horizon=24):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        
        self.feature_mins = self.data.min(axis=0).values
        self.feature_maxs = self.data.max(axis=0).values
    
    def __len__(self):
        return len(self.data) - (self.window_size + self.horizon) + 1

    def __getitem__(self, idx):
        window = self.data[idx: idx + self.window_size + self.horizon]
        past_all_features = window[:self.window_size]
        future_energy = window[self.window_size:, -1:]
        
        scaled_past_all_features = (past_all_features - self.feature_mins) / (self.feature_maxs - self.feature_mins)
        
        scaled_future_energy, energy_min, energy_max = minmax_scale(future_energy)

        return scaled_past_all_features, scaled_future_energy, (energy_min, energy_max)

def swish(x):
    return x * torch.sigmoid(x)

 def polynomial_kernel(self, x):
   # Polynomial kernel equation
   return (self.alpha * torch.mm(x, self.references.t()) + self.coef0) ** self.degree

def rbf_kernel(self, x):
    # RBF kernel equation
    x = x.unsqueeze(1) - self.references.unsqueeze(0)
    return torch.exp(-self.gamma * (x ** 2).sum(dim=2))

def combined_kernel(self, x):
   return self.weight * self.rbf_kernel(x) + (1 - self.weight) * self.polynomial_kernel(x)
    
