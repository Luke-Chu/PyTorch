import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

# writer.add_image()
# writer.add_scalar()

for i in range(100):
    writer.add_scalar("y=x^2", i*i, i)

img_pil = Image.open("../hymenoptera_data/train/ants/6240329_72c01e663e.jpg")
img_array = np.array(img_pil)

writer.add_image("ants", img_array, 3, dataformats='HWC')


writer.close()
