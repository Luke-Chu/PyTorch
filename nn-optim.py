import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.seq(x)


model = Model()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 一轮代表一次学习，实际训练中往往成百上千轮
for epoch in range(20):
    running_loss = 0.0  # 可以打印每轮学习的总误差
    for data in dataloader:
        images, targets = data
        out = model(images)
        if epoch == 0:
            print(out[:10])
            print(targets[:10])
        if epoch == 5:
            print(out[:10])
            print(targets[:10])
        # 优化步骤
        result_loss = loss(out, targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        # 将每组误差加和
        running_loss = running_loss + result_loss
    # 打印每轮学习误差，可以看见总误差是不断减小的
    print(running_loss)
