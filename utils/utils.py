import torch
import numpy as np
import cv2 as cv
from skimage import io
from PIL import Image
import os
import subprocess

MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def array_to_heatmap(x):
    x = (x - x.min()) / (x.max() - x.min()) * 255
    x = x.astype(np.uint8)
    return cv.applyColorMap(x.astype(np.uint8), cv.COLORMAP_RAINBOW)

def img_to_tensor(img_path, device, size=None, mode='rgb'):
    """
    Read image from img_path, and convert to (C, H, W) tensor in range [-1, 1]
    """
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    if mode=='bgr':
        img = img[..., ::-1]
    if size:
        img = cv.resize(img, size)
    img = img / 255 * 2 - 1 
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device) 
    return img_tensor.float()

def tensor_to_img(tensor, save_path=None, size=None, mode='RGB', normal=[-1, 1]):
    """
    mode: RGB or L (gray image)
    Input: tensor with shape (C, H, W)
    Output: PIL Image
    """
    if isinstance(size, int):
        size = (size, size)
    img_array = tensor.squeeze().data.cpu().numpy()
    if mode == 'RGB':
        img_array = img_array.transpose(1, 2, 0)

    if size is not None:
        img_array = cv.resize(img_array, size, interpolation=cv.INTER_LINEAR)

    if len(normal):
        img_array = (img_array - normal[0]) / (normal[1] - normal[0]) * 255
        img_array = img_array.clip(0, 255)

    img_array = img_array.astype(np.uint8)
    if save_path:
        img = Image.fromarray(img_array, mode)
        img.save(save_path)

    return img_array

def tensor_to_numpy(tensor):
    return tensor.data.cpu().numpy()

def batch_numpy_to_image(array, size=None):
    """
    Input: numpy array (B, C, H, W) in [-1, 1]
    """
    if isinstance(size, int):
        size = (size, size)

    out_imgs = []
    array = np.clip((array + 1)/2 * 255, 0, 255) 
    array = np.transpose(array, (0, 2, 3, 1))
    for i in range(array.shape[0]):
        if size is not None:
            tmp_array = cv.resize(array[i], size)
        else:
            tmp_array = array[i]
        out_imgs.append(tmp_array)
    return np.array(out_imgs).astype(np.uint8)

def batch_tensor_to_img(tensor, size=None):
    """
    Input: (B, C, H, W) 
    Return: RGB image, [0, 255]
    """
    arrays = tensor_to_numpy(tensor)
    out_imgs = batch_numpy_to_image(arrays, size)
    return out_imgs 

def color_parse_map(tensor, size=None):
    """
    input: tensor or batch tensor
    return: colorized parsing maps
    """
    if len(tensor.shape) < 4:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] > 1:
        tensor = tensor.argmax(dim=1) 

    tensor = tensor.squeeze(1).data.cpu().numpy()
    color_maps = []
    for t in tensor:
        tmp_img = np.zeros(tensor.shape[1:] + (3,))        
        for idx, color in enumerate(MASK_COLORMAP):
            tmp_img[t == idx] = color
        if size is not None:
            tmp_img = cv.resize(tmp_img, (size, size))
        color_maps.append(tmp_img.astype(np.uint8))
    return color_maps

def onehot_parse_map(img):
    """
    input: RGB color parse map
    output: one hot encoding of parse map
    """
    n_label = len(MASK_COLORMAP)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[:2]
    onehot_label = np.zeros((n_label, h, w))
    colormap = np.array(MASK_COLORMAP).reshape(n_label, 1, 1, 3)
    colormap = np.tile(colormap, (1, h, w, 1))
    for idx, color in enumerate(MASK_COLORMAP):
        tmp_label = colormap[idx] == img
        onehot_label[idx] = tmp_label[..., 0] * tmp_label[..., 1] * tmp_label[..., 2]
    return onehot_label
 

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def get_gpu_memory_map():
    """Get the current gpu usage within visible cuda devices.

    Returns
    -------
    Memory Map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    Device Ids: gpu ids sorted in descending order according to the available memory.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = sorted([int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')])
    else: 
        visible_devices = range(len(gpu_memory))
    gpu_memory_map = dict(zip(range(len(visible_devices)), gpu_memory[visible_devices]))
    return gpu_memory_map, sorted(gpu_memory_map, key=gpu_memory_map.get)


if __name__ == '__main__':
    hm = torch.randn(32, 68, 128, 128).cuda()
    flip(hm, 2)
    x = torch.ones(32, 68)
    y = torch.ones(32, 68)
    print(get_gpu_memory_map())



