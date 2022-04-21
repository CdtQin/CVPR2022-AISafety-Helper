import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fake_params = nn.Conv2d(8, 8, 3)

    def forward(self, x):
        return torch.randn(x.shape[0], 1)