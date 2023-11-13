import torch
import torchvision
from PIL import Image
from torchvision import transforms

from model import Model


image_path = "../imgs/ship.png"
image = Image.open(image_path).convert("RGB")

transform = torchvision.transforms.Compose([transforms.Resize([32, 32]),
                                            transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, [1, 3, 32, 32])

print(image.shape)


model = Model()
model.load_state_dict(torch.load("model_30.pth"))
# print(model)

# 开始测试
model.eval()
with torch.no_grad():
    out = model(image)
    # print(out)
    print(out.argmax(1))
