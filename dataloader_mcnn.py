from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

class CrowdDataset(Dataset):

    def __init__(self,img_root,gt_map_path,gt_downsample=1):

        self.img_root=img_root
        self.gt_map_path=gt_map_path
        self.gt_downsample=gt_downsample

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        print(img.dtype)
        gt_dmap=np.load(os.path.join(self.gt_map_path,img_name.replace('.jpg','.npz')))['arr']
        gt_dmap = gt_dmap.astype(np.float32)
        if self.gt_downsample>1:
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1))
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*(self.gt_downsample**2)
            #gt_dmap = gt_dmap.transpose((1,2,0))
        img_tensor=torch.tensor(img/255,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        print(img_tensor.shape)
        print(gt_dmap_tensor.shape)
        return img_tensor,gt_dmap_tensor