import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# input = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
# print(input.shape)
# target = torch.tensor(1)
# loss = nn.CrossEntropyLoss()(input, target)
# print(loss)  # -0.3+ln(exp(0.2)+exp(0.3)+exp(0.4))


dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


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
for data in dataloader:
    images, targets = data
    out = model(images)
    print(out)
    print(targets)
    result_loss = loss(out, targets)
    result_loss.backward()
    print(result_loss)

