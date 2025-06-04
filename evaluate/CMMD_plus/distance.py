"""Memory-efficient MMD implementation."""

import torch
import numpy as np

_SIGMA = 10
_SCALE = 1000

def mmd(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def single_image_mmd(x_single, y, variance=0.01):
    x_single = torch.from_numpy(x_single)
    y = torch.from_numpy(y)
    
    # 构建分布
    n = y.shape[0]
    noise = torch.randn(n, x_single.shape[1]) * np.sqrt(variance)
    x = x_single.repeat(n, 1) + noise
    
    return mmd(x.numpy(), y.numpy())