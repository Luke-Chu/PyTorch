import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.seq(x)


# 给一个输入，看是否正确输出，可测试模型是否正确
if __name__ == '__main__':
    input = torch.ones(64, 3, 32, 32)
    model = Model()
    output = model(input)
    print(output)
