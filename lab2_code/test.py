import torch.utils.data as data
import torch
import os
import PIL.Image as Image
from torchvision.transforms import transforms as T
import rasterio
root1="picData2/manual1"
root2="picData2/images"
imgs1=[]
for fileName in os.listdir(r"./"+root1):
    print(fileName)
    if(fileName[-1]!='f'):
        continue
    
    img=os.path.join(root1,fileName)
    img_y=rasterio.open(img)
    img_y=img_y.read()
    print(img_y)
    img_y=T.ToTensor()(img_y)
    print(img_y)
    img_y=img_y.reshape((1,2336,3504))
    print(img_y.shape)
    print(torch.max(img_y))
    break
    imgs1.append(img)
for fileName in os.listdir(r"./"+root2):
    img=os.path.join(root2,fileName)
    img_y=Image.open(img)
    #img_y=img_y.read()
    print(img_y)
    img_y=T.ToTensor()(img_y)
    print(img_y)
    print(img_y.shape)
    print(torch.max(img_y))
    break