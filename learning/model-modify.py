import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# train_data = torchvision.datasets.ImageNet("./datasets", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())


vgg16 = torchvision.models.vgg16(weights='DEFAULT')
print(vgg16)

# vgg16.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg16)
# vgg16.classifier.add_module("7", nn.Linear(1000, 10))
# print(vgg16)
vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16)
