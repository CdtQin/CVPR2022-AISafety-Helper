import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.load_params()

    def load_params(self):
        # import os
        # os.path.join(os.path.dirname(__file__), 'ckpt.pth')
        pass

    def forward(self, x):
        pass
