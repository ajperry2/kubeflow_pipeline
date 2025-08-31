import torch
import torch.nn


class LayerNormalization(torch.nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
