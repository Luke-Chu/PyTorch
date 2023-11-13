import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


writer = SummaryWriter("logs")

img_path = "D:\\Users\\Luke\\Pictures\\博士论坛.png"
img_pil = Image.open(img_path)

img_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
img_tensor = img_trans(img_pil)

writer.add_image("tensor", img_tensor)
trans_norm = v2.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# Resize使用
trans_resize = v2.Resize([512, 512], antialias=True)
img_resize = trans_resize(img_tensor)
writer.add_image("Resize", img_resize, 0)

# Compose使用
trans_com = v2.Compose([v2.Resize(128), v2.ToTensor()])
img_com = trans_com(img_pil)
writer.add_image("Compose", img_com, 2)

# RandomCrop使用
trans_crop = v2.RandomCrop([256, 512])
for i in range(10):
    img_crop = trans_crop(img_tensor)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
