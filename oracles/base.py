import torch
from torch import nn

class BaseOracle(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)