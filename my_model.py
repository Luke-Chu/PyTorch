import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1


module = MyModule()
x = torch.tensor(1.0)
print(type(x))
out = module(x)

print(out)
