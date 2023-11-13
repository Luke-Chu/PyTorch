import torch
from torch import nn

_input = torch.tensor(2)
# _input = torch.reshape(_input, [-1, 1, 2, 2])
print(_input.shape)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.sigmoid1 = nn.Sigmoid()
        self.norm1 = nn.BatchNorm2d()

    def forward(self, _input):
        # return self.relu1(_input)
        return self.sigmoid1(_input)


model = Model()
out = model(_input)
print(out)
