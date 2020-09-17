import os
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils

class EnhanceModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--lambda_pix', type=float, default=10.0, help='weight for parsing map')
            parser.add_argument('--lambda_pcp', type=float, default=1.0, help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_fm', type=float, default=10.0, help='weight for sr')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for sr')
            parser.add_argument('--lambda_s', type=float, default=100., help='weight for global style')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netP = networks.define_P(opt)
        self.netG = networks.define_G(opt, use_norm='spectral_norm')

        if self.isTrain:
            self.netD = networks.define_D(opt, opt.Dinput_nc, use_norm='spectral_norm') 
            self.vgg_model = loss.PCPFeat(weight_path='./pretrain_models/vgg19-dcbb9e9d.pth').to(opt.data_device)

        self.model_names = ['G']
        self.loss_names = ['Pix', 'PCP', 'G', 'FM', 'D', 'S'] # Generator loss, fm loss, parsing loss, discriminator loss
        self.visual_names = ['img_LR', 'img_HR', 'img_SR', 'ref_Parse']
        self.fm_weights = [1**x for x in range(opt.D_num)]

        if self.isTrain:
            self.model_names = ['G', 'D']

            self.criterionParse = torch.nn.CrossEntropyLoss().to(opt.data_device)
            self.criterionFM = loss.FMLoss().to(opt.data_device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix= nn.L1Loss()
            self.criterionRS = loss.RegionStyleLoss()

            self.optimizer_G = optim.Adam([p for p in self.netG.parameters() if p.requires_grad], lr=opt.lr/2, betas=(opt.beta1, 0.999))
            self.optimizer_D = optim.Adam([p for p in self.netD.parameters() if p.requires_grad], lr=opt.lr*2, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def load_pretrain_models(self,):
        self.netP.eval()
        print('Loading pretrained LQ face parsing network from', self.opt.parse_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netP.module.load_state_dict(torch.load(self.opt.parse_net_weight))
        else:
            self.netP.load_state_dict(torch.load(self.opt.parse_net_weight))
        self.netG.eval()
        print('Loading pretrained PSFRGAN from', self.opt.psfr_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netG.module.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)
        else:
            self.netG.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)
    
    def set_input(self, input):
        self.img_LR = input['LR'].to(self.opt.data_device)
        self.img_HR = input['HR'].to(self.opt.data_device)
        if self.opt.debug:
            print('SRNet input shape:', self.img_LR.shape, self.img_HR.shape)

    def forward(self):
        with torch.no_grad():
            ref_mask, _ = self.netP(self.img_LR) 
            self.ref_mask_onehot = (ref_mask == ref_mask.max(dim=1, keepdim=True)[0]).float().detach()

            hr_ref_mask, _ = self.netP(self.img_HR) 
            self.hr_ref_mask_onehot = (hr_ref_mask == hr_ref_mask.max(dim=1, keepdim=True)[0]).float().detach()
        if self.opt.debug:
            print('SRNet reference mask shape:', self.ref_mask_onehot.shape)
        self.pred_mask, self.img_SR = self.netG(self.img_LR, self.ref_mask_onehot) 

        self.real_D_results = self.netD(torch.cat((self.img_HR, self.hr_ref_mask_onehot), dim=1), return_feat=True)
        self.fake_D_results = self.netD(torch.cat((self.img_SR.detach(), self.hr_ref_mask_onehot), dim=1), return_feat=False)
        self.fake_G_results = self.netD(torch.cat((self.img_SR, self.hr_ref_mask_onehot), dim=1), return_feat=True)

        self.img_SR_feats = self.vgg_model(self.img_SR)
        self.img_HR_feats = self.vgg_model(self.img_HR)

    def backward_G(self):
        # Pix Loss
        self.loss_Pix = self.criterionPix(self.img_SR, self.img_HR) * self.opt.lambda_pix    

        # Region style loss
        self.loss_RS = self.criterionRS(self.img_SR_feats, self.img_HR_feats, self.hr_ref_mask_onehot) * self.opt.lambda_rs

        # perceptual loss
        self.loss_PCP = self.criterionPCP(self.img_SR_feats, self.img_HR_feats) * self.opt.lambda_pcp

        # Feature matching loss
        tmp_loss =  0
        for i, w in zip(range(self.opt.D_num), self.fm_weights):
            tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) * w
        self.loss_FM = tmp_loss * self.opt.lambda_fm / self.opt.D_num

        # Generator loss
        tmp_loss = 0
        for i in range(self.opt.D_num):
            tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
        self.loss_G = tmp_loss * self.opt.lambda_g / self.opt.D_num
        
        total_loss = self.loss_Pix + self.loss_PCP + self.loss_FM + self.loss_G + self.loss_RS        
        total_loss.backward()

    def backward_D(self, ):
        self.loss_D = 0
        for i in range(self.opt.D_num):
            self.loss_D += 0.5 * (self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0], True))
        
        self.loss_D.backward()
    
    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D ------------
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_visuals(self, size=512):
        out = []
        visual_imgs = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        out_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        visual_imgs.append(out_imgs[0])
        visual_imgs.append(utils.color_parse_map(self.ref_mask_onehot))
        visual_imgs.append(out_imgs[1])
        visual_imgs.append(out_imgs[2])
        visual_imgs.append(utils.color_parse_map(self.hr_ref_mask_onehot))

        return visual_imgs

