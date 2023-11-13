import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# train_dataset = torchvision.datasets.CIFAR10("./datasets", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transforms.ToTensor(), download=True)

# img, target = test_dataset[0]
# img.show()
# print(target)
# print(test_dataset[0])

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    batch_step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, batch_step)
        batch_step += 1

writer.close()
