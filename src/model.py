import json
import torch
from torch import nn

class ModelArgs:

    dim: int

    def __init__(self):
        try:
            with open('modelconfig.json', 'r') as file:
                data = json.load(file)
                data = data['dim']
        except json.JSONDecodeError:
            print('config.json is not a valid JSON file')


class RMSNorm(nn.Module):
    """
    Root-mean-square normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x ):
        return nn.functional.rms_norm(x, (self.dim,),self.weight, self.eps)

class Transformer:
    def __init__(self):
        pass

class Model:
    def __init__(self):
        pass
