import torch
import torch.nn as nn
import torch.nn.functional as F


# DropBlock (более агрессивный аналог Dropout для ConvNet)
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.3, block_size=5):  # Увеличьте drop_prob
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :, ::self.block_size, ::self.block_size]) < gamma).float()
        mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        return x * (1 - mask)