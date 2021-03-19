import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

from data.image_folder import make_dataset

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from utils.utils import onehot_parse_map

from data.ffhq_dataset import complex_imgaug, random_gray

class CelebAHQMaskDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.Pimg_size
        self.lr_size = opt.Gin_size
        self.hr_size = opt.Gout_size
        self.shuffle = True if opt.isTrain else False 

        self.img_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'CelebA-HQ-img')))
        self.mask_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'CelebAMask-HQ-mask')))

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __len__(self,):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        sample = {}
        img_path = self.img_dataset[idx]
        mask_path = self.mask_dataset[idx]
        hr_img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path)
                    
        hr_img = hr_img.resize((self.hr_size, self.hr_size))
        hr_img = random_gray(hr_img, p=0.3)
        scale_size = np.random.randint(32, 256)
        lr_img = complex_imgaug(hr_img, self.img_size, scale_size)

        mask_img = mask_img.resize((self.hr_size, self.hr_size))
        mask_label = torch.tensor(np.array(mask_img)).long()
 
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path, 'Mask': mask_label}



