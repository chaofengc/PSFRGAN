import argparse
import os
import numpy as np
import random
from utils import utils
import torch
import models
import data
from utils import utils


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=False, help='path to images')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpus', type=int, default=1, help='how many gpus to use')
        parser.add_argument('--seed', type=int, default=123, help='Random seed for training')
        parser.add_argument('--checkpoints_dir', type=str, default='./check_points', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='enhance', help='chooses which model to train [parse|enhance]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--Dinput_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=4, help='downsampling layers in discriminator')
        parser.add_argument('--D_num', type=int, default=3, help='numbers of discriminators')

        parser.add_argument('--Pnorm', type=str, default='bn', help='parsing net norm [in | bn| none]')
        parser.add_argument('--Gnorm', type=str, default='spade', help='generator norm [in | bn | none]')
        parser.add_argument('--Dnorm', type=str, default='in', help='discriminator norm [in | bn | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--dataset_name', type=str, default='single', help='dataset name')
        parser.add_argument('--Pimg_size', type=int, default='512', help='image size for face parse net')
        parser.add_argument('--Gin_size', type=int, default='512', help='image size for face parse net')
        parser.add_argument('--Gout_size', type=int, default='512', help='image size for face parse net')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        parser.add_argument('--debug', action='store_true', help='if specified, set to debug mode')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        opt.expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(opt.expr_dir)
        file_name = os.path.join(opt.expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        opt.log_dir = os.path.join(opt.checkpoints_dir, 'log_dir')
        utils.mkdirs(opt.log_dir)
        opt.log_archive = os.path.join(opt.checkpoints_dir, 'log_archive')
        utils.mkdirs(opt.log_archive)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        if opt.debug:
            opt.name = 'debug'
            opt.save_iter_freq = 1
            opt.save_latest_freq = 1
            opt.visual_freq = 1
            opt.print_freq = 1

        # Find avaliable GPUs automatically
        if opt.gpus > 0:
            opt.gpu_ids = utils.get_gpu_memory_map()[1][:opt.gpus]
            if not isinstance(opt.gpu_ids, list):
                opt.gpu_ids = [opt.gpu_ids]
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0 % opt.gpus]))
            opt.data_device = torch.device('cuda:{}'.format(opt.gpu_ids[1 % opt.gpus]))
        else:
            opt.gpu_ids = []
            opt.device = torch.device('cpu')

        # set random seeds to ensure reproducibility
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt
