import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  unet import U_Net
from unet import init_weights
import tifffile as tiff
import cv2
import numpy as np
def getData2(filePath):
    ans=[]
    for fileName in os.listdir(r"./"+filePath):
        gif=cv2.VideoCapture(fileName)
        _,frame=gif.read()
        gif.release()
        #img=cv2.imread(filePath+"/"+fileName)
        ans.append(frame)
    return ans  
def getData(filePath):
    ans=[]
    for fileName in os.listdir(r"./"+filePath):
        if(fileName=="Thumbs.db"):
            continue
        img=cv2.imread(filePath+"/"+fileName)
        ans.append(img)
        print(fileName)
        print(img.shape)
    return ans
def get_k_fold_data(k,i,X,Y):
    fold_size=X.shape[0]//k
    x_train,y_train=None, None
    for j in range(k):
        start_index=j*fold_size
        end_index=(j+1)*fold_size
        x_part=X[start_index:end_index,:]
        y_part=Y[start_index:end_index]
        if(j==i):
            #put it into train
            x_valid=x_part
            y_valid=y_part
        elif x_train is None:
            x_train=x_part
            y_train=y_part
        else:
            x_train=torch.concatenate((x_train,x_part),axis=0)
            y_train=torch.concatenate((y_train,y_part),axis=0)
    return x_train,y_train,x_valid,y_valid
loss=nn.BCELoss()   
def entropyLoss(net,featrues,labels):
    y=net(featrues)
    
def train(net,x_train,y_train,x_valid,y_valid,lr,epoch_num):
    train_ls,test_ls=[],[]
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    for epoch in range(epoch_num):
        for X,y in (x_train,y_train):
            optimizer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        
def entry(epoch_num,k,lr):
    net=U_Net()
    init_weights(net)
    

    dataPath1="lab2_code/picData2/images"
    dataPath2="lab2_code/picData2/manual1"

    
    X=torch.tensor(np.array(getData(dataPath1),dtype=float))
    print(X.shape)
    Y=torch.tensor(np.array(getData(dataPath2),dtype=float))
    print(X.shape)
    for i in range(k):
        x_train,y_train,x_valid,y_valid=get_k_fold_data(k,i,X,Y)
        train_ls,valid_ls=train(net,x_train,y_train,x_valid,y_valid,lr,epoch_num)
