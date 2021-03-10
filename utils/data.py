# -*- coding: utf-8 -*-
import os, scipy.misc
from glob import glob
import numpy as np 
import h5py

class CelebA():
    def __init__(self, datapath):
        self.resolution = [
                        'data2x2', 'data4x4', 'data8x8', 'data16x16', 
                        'data32x32', 'data64x64', 'data128x128', 'data256x256', 
                        'data512x512', 'data1024x1024'
                        ]
        self._base_key = 'data'
        self.dataset = h5py.File(datapath, 'r')
        self._len = {k:len(self.dataset[k]) for k in self.resolution}
        assert all([resol in self.dataset.keys() for resol in self.resolution])

    def __call__(self, batch_size, resols):
        idx = np.random.randint(self._len[self.resolution[0]], size=batch_size)
        batch_out_all = {} 
        for r in resols:
            key = 'data{}x{}'.format(r, r)
            batch_x = np.array([self.dataset[key][i]/127.5-1.0 for i in idx], dtype=np.float32)
            batch_out_all[r] = batch_x
        return batch_out_all

    def save_imgs(self, samples, file_name):
        N_samples, channel, height, width = samples.shape
        N_row = N_col = int(np.ceil(N_samples**0.5))
        combined_imgs = np.ones((channel, N_row*height, N_col*width))
        for i in range(N_row):
            for j in range(N_col):
                if i*N_col+j < samples.shape[0]:
                    combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
        combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
        scipy.misc.imsave(file_name+'.png', combined_imgs)


def get_img(img_path, is_crop=True, crop_h=256, resize_h=64, normalize=False):
    img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    resize_w = resize_h
    if is_crop:
        crop_w = crop_h
        h, w = img.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
    else:
        cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
    if normalize:
        cropped_image = cropped_image/127.5 - 1.0
    return np.transpose(cropped_image, [2, 0, 1])



class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian'):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        self.generator_map = {'gaussian': np.random.randn, 'uniform': np.random.uniform}
        if self.noise_type == 'gaussian':
            self.generator = lambda s: np.random.randn(*s)
        elif self.noise_type == 'uniform':
            self.generator = lambda s: np.random.uniform(-1, 1, size=s)

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)
