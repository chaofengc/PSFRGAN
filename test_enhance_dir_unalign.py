'''
This script enhance images with unaligned faces in a folder and paste it back to the original place.
'''
import dlib
import os
import cv2
import numpy as np 
from tqdm import tqdm
from skimage import transform as trans
from skimage import io

import torch
from utils import utils
from options.test_options import TestOptions
from models import create_model

from test_enhance_single_unalign import * 


if __name__ == '__main__':
    opt = TestOptions().parse()
    #  face_detector = dlib.get_frontal_face_detector()
    face_detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    lmk_predictor = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')
    template_path = './pretrain_models/FFHQ_template.npy'
    enhance_model = def_models(opt)

    for img_name in os.listdir(opt.src_dir):
        img_path = os.path.join(opt.src_dir, img_name)
        save_current_dir = os.path.join(opt.results_dir, os.path.splitext(img_name)[0])
        os.makedirs(save_current_dir, exist_ok=True)
        print('======> Loading image', img_path)
        img = dlib.load_rgb_image(img_path)
        aligned_faces, tform_params = detect_and_align_faces(img, face_detector, lmk_predictor, template_path)
        # Save aligned LQ faces
        save_lq_dir = os.path.join(save_current_dir, 'LQ_faces') 
        os.makedirs(save_lq_dir, exist_ok=True)
        print('======> Saving aligned LQ faces to', save_lq_dir)
        save_imgs(aligned_faces, save_lq_dir)

        hq_faces, lq_parse_maps = enhance_faces(aligned_faces, enhance_model)
        # Save LQ parsing maps and enhanced faces
        save_parse_dir = os.path.join(save_current_dir, 'ParseMaps') 
        save_hq_dir = os.path.join(save_current_dir, 'HQ') 
        os.makedirs(save_parse_dir, exist_ok=True)
        os.makedirs(save_hq_dir, exist_ok=True)
        print('======> Save parsing map and the enhanced faces.')
        save_imgs(lq_parse_maps, save_parse_dir)
        save_imgs(hq_faces, save_hq_dir)

        print('======> Paste the enhanced faces back to the original image.')
        hq_img = past_faces_back(img, hq_faces, tform_params, upscale=opt.test_upscale) 
        final_save_path = os.path.join(save_current_dir, 'hq_final.jpg') 
        print('======> Save final result to', final_save_path)
        io.imsave(final_save_path, hq_img)


