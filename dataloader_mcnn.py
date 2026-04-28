from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
from PIL import Image

class CrowdDataset(Dataset):

    def __init__(self,img_root,gt_map_path,gt_downsample=1, resize=False, transform=None, patch = False,
                 train = False):

        self.img_root=img_root
        self.gt_map_path=gt_map_path
        self.gt_downsample=gt_downsample

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)
        self.resize = resize
        self.transform = transform
        self.patch = patch
        self.train = train

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gt_dmap=np.load(os.path.join(self.gt_map_path,img_name.replace('.jpg','.npz')))['arr']
        gt_dmap = gt_dmap.astype(np.float32)

        if self.patch:
            img = cv2.resize(img, (1152, 768))

            gt_count = np.sum(gt_dmap)

            # augmentation
            if self.train and random.random() > 0.5:
                img = Image.fromarray(img)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)   # (C, H, W)

            C, H, W = img.shape

            ph, pw = 384, 384
            m = W // pw   # 3
            n = H // ph   # 2

            patches = []

            for i in range(m):
                for j in range(n):

                    x1 = i * pw
                    x2 = (i + 1) * pw
                    y1 = j * ph
                    y2 = (j + 1) * ph

                    patch = img[:, y1:y2, x1:x2]
                    patches.append(patch)

            img_return = torch.stack(patches) 

            return self.img_root, img_return, gt_count
        if self.resize:
            orig_h, orig_w = img.shape[:2]
            TARGET_SIZE = (orig_w//128*128, orig_h//128*128)
            img = cv2.resize(img, TARGET_SIZE)
            gt_dmap = cv2.resize(gt_dmap, TARGET_SIZE[::-1])
            gt_dmap = gt_dmap * ((orig_h * orig_w) / (TARGET_SIZE[1] * TARGET_SIZE[0]))
        if self.gt_downsample>1:
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*(self.gt_downsample**2)
            #gt_dmap = gt_dmap.transpose((1,2,0))
        
        img=img.transpose((2,0,1))
        img_tensor=torch.tensor(img/255,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor,gt_dmap_tensor