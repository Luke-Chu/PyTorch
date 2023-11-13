import torch
from torch import nn


_input = torch.tensor([[2, 3],
                       [1, 4]], dtype=torch.float32)
print(_input)
out1 = torch.flatten(_input)
print(out1)
out = nn.Linear(in_features=4, out_features=2, bias=False)(out1)
print(out)
