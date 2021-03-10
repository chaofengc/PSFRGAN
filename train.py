from utils.timer import Timer
from utils.logger import Logger
from utils import utils

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

import torch 
import os
import torch.multiprocessing as mp

def train(opt):
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
        
    logger = Logger(opt)
    timer = Timer()

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.total_epochs + 1):    
        for i, data in enumerate(dataset, start=start_iter):
            cur_iters += 1
            logger.set_current_iter(cur_iters)
            # =================== load data ===============
            model.set_input(data, cur_iters)
            timer.update_time('DataTime')
    
            # =================== model train ===============
            model.forward(), timer.update_time('Forward')
            model.optimize_parameters() 
            loss = model.get_current_losses()
            loss.update(model.get_lr())
            logger.record_losses(loss)
            timer.update_time('Backward')

            # =================== save model and visualize ===============
            if cur_iters % opt.print_freq == 0:
                print('Model log directory: {}'.format(opt.expr_dir))
                epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
                logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)
    
            if cur_iters % opt.visual_freq == 0:
                visual_imgs = model.get_current_visuals()
                logger.record_images(visual_imgs)
            
            if cur_iters % opt.save_iter_freq == 0:
                print('saving current model (epoch %d, iters %d)' % (epoch, cur_iters))
                save_suffix = 'iter_%d' % cur_iters 
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks(save_suffix, info)

            if cur_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, iters %d)' % (epoch, cur_iters))
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks('latest', info)

            if i >= single_epoch_iters - 1:
                start_iter = 0
                break

            #  model.update_learning_rate()
            if opt.debug: break
        if opt.debug and epoch >= 0: break 
    logger.close()

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
    
