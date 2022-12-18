import unet
import os
import torch
import cv2
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import dataset


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = unet.UNet().to(device)
    net.load_state_dict(torch.load('model_100_0.053562819957733154.plt'))
    loader = DataLoader(dataset.Datasets(r"DRIVE\test"),
                        batch_size=1, shuffle=True, num_workers=2)
    epoch = 1
    for inputs, labels in loader:
        # 图片和分割标签
        inputs, labels = inputs.to(device), labels.to(device)
        # 输出生成的图像
        out = net(inputs)
        x = inputs[0]
        # 生成的图像，取第一张
        x_ = out[0]
        # 标签的图像，取第一张
        y = labels[0]
        # 三张图，从第0轴拼接起来，再保存
        img = torch.stack([x, x_, y], 0)
        save_image(img.cpu(), os.path.join(r'./test_img', f"{epoch}.png"))
        epoch+=1
    
