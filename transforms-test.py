from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/9715481_b3cb4114ff.jpg"
img_pil = Image.open(img_path)

img_trans = transforms.ToTensor()
img_tensor = img_trans(img_pil)

# print(img_tensor)

# 转换后就可以直接使用tensorboard的add_image()函数了
writer = SummaryWriter("logs")

# 添加图像
writer.add_image("tensor-img", img_tensor)

writer.close()
