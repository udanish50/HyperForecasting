import torch

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def mae_loss(y_pred, y_true):
    return torch.abs(y_pred - y_true).mean()

