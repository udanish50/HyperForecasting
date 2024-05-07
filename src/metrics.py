import os
import torch
    
def smape(y_true, y_pred):
    return 100 * torch.mean(2 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred))).item()

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))
