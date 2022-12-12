import torch.utils.data as data
import os
import PIL.Image as Image
from torchvision.transforms import transforms as T
import rasterio
#data.Dataset:
#所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
 
class LiverDataset(data.Dataset):
    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,root1,root2,transform = None,target_transform = None):#root表示图片路径
        #n = len(os.listdir(root1))#os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        #m=len(os.listdir(root2))
        imgs = []
        imgs1=[]
        imgs2=[]
        for fileName in os.listdir(r"./"+root1):
            if(fileName[-1]!='g' and fileName[-1]!='G'):
                continue
            print(fileName)
            img=os.path.join(root1,fileName)
            imgs1.append(img)

        for fileName in os.listdir(r"./"+root2):
            if(fileName[-1]!='f'):
                continue
            print(fileName)
            img=os.path.join(root2,fileName)
            imgs2.append(img)
        
        n=len(imgs1)
        for i in range(n):
            imgs.append([imgs1[i],imgs2[i]])

        '''for i in range(0,n,3):
            if(i<10):
                name="0"+str(i)
            lastName=["dr","g","h"]
            for last in lastName:
                img = os.path.join(root1,"%03d.png"%i)#os.path.join(path1[,path2[,......]]):将多个路径组合后返回
                mask = os.path.join(root,"%03d_mask.png"%i)
                imgs.append([img,mask])#append只能有一个参数，加上[]变成一个list
        '''    
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        
        img_x = Image.open(x_path)
        
        img_y = rasterio.open(y_path)
        img_y=img_y.read()
        
        
        if self.transform is not None:
            img_x = self.transform(img_x)
            #print(img_x)
        if self.target_transform is not None:
            #print("process Y before")
            #img_y=T.ToTensor()(img_y)
            img_y = self.target_transform(img_y)
            #print("process Y")
            #print(img_y)
            #print(img_y)
            img_y=img_y.reshape((1,2336,3504))
            print(img_y.shape)
        return img_x,img_y#返回的是图片
    
    
    def __len__(self):
        return len(self.imgs)#400,list[i]有两个元素，[img,mask]