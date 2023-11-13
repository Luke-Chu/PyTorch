import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 定义神经网络模块
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, _input):
        return self.pool1(_input)


model = Model()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    images, targets = data
    writer.add_images("origin_images", images, step)
    out = model(images)
    writer.add_images("max_pool", out, step)
    step += 1

writer.close()
