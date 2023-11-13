import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


# 定义神经网络模块
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, _input):
        return self.conv1(_input)


model = Model()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    out = model(imgs)
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    print(out.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("images", imgs, step)

    # tensorboard无法显示6个通道的图像，所以将其转化为3个通道（该方法并不严谨）
    out = torch.reshape(out, [-1, 3, 30, 30])
    print(out.shape)  # torch.Size([128, 3, 30, 30])
    writer.add_images("conv2d", out, step)
    step += 1

