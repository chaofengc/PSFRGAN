import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time 
import numpy as np

if __name__ == '__main__':
    opt = TestOptions()
    opt = opt.parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.load_pretrain_models()

    netP = model.netP
    model.eval()
    for i, data in tqdm(enumerate(dataset), total=len(dataset)//opt.batch_size):
        inp = data['LR']
        with torch.no_grad():
            parse_map, _ = netP(inp)
            parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
        img_path = data['LR_paths']     # get image paths
        ref_parse_img = utils.color_parse_map(parse_map_sm)
        for i in range(len(img_path)):
            save_path = os.path.join(opt.save_masks_dir, os.path.basename(img_path[i]))
            os.makedirs(opt.save_masks_dir, exist_ok=True)
            save_img = Image.fromarray(ref_parse_img[i])
            save_img.save(save_path)


       
 
