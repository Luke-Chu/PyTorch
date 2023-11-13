import torch
import torchvision

vgg16 = torchvision.models.vgg16()
torch.save(vgg16, "vgg16_method1.pth")

torch.save(vgg16.state_dict(), "vgg16_method2.pth")
