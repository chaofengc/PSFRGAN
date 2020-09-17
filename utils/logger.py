import os
from collections import OrderedDict
import numpy as np
from .utils import mkdirs
from tensorboardX import SummaryWriter
from datetime import datetime
import socket
import shutil

class Logger():
    def __init__(self, opts):
        time_stamp = '_{}'.format(datetime.now().strftime('%Y-%m-%d_%H:%M'))
        self.opts = opts
        self.log_dir = os.path.join(opts.log_dir, opts.name+time_stamp)
        self.phase_keys = ['train', 'val', 'test']
        self.iter_log = []
        self.epoch_log = OrderedDict() 
        self.set_mode(opts.phase)

        # check if exist previous log belong to the same experiment name 
        exist_log = None 
        for log_name in os.listdir(opts.log_dir):
            if opts.name in log_name:
                exist_log = log_name 
        if exist_log is not None: 
            old_dir = os.path.join(opts.log_dir, exist_log)
            archive_dir = os.path.join(opts.log_archive, exist_log) 
            shutil.move(old_dir, archive_dir)

        self.mk_log_file()

        self.writer = SummaryWriter(self.log_dir)

    def mk_log_file(self):
        mkdirs(self.log_dir)
        self.txt_files = OrderedDict()
        for i in self.phase_keys:
            self.txt_files[i] = os.path.join(self.log_dir, 'log_{}'.format(i))

    def set_mode(self, mode):
        self.mode = mode
        self.epoch_log[mode] = []

    def set_current_iter(self, cur_iter):
        self.cur_iter = cur_iter
        
    def record_losses(self, items):
        """
        iteration log: [iter][{key: value}]
        """
        self.iter_log.append(items)
        for k, v in items.items():
            if 'loss' in k.lower():
                self.writer.add_scalar('loss/{}'.format(k), v, self.cur_iter)

    def record_scalar(self, items):
        """
        Add scalar records. item, {key: value}
        """
        for i in items.keys():
            self.writer.add_scalar('{}'.format(i), items[i], self.cur_iter)

    def record_images(self, visuals, nrow=6, tag='ckpt_image'):
        imgs = []
        max_len = visuals[0].shape[0] 
        for i in range(nrow):
            if i >= max_len: continue
            tmp_imgs = [x[i] for x in visuals]
            imgs.append(np.hstack(tmp_imgs))
        imgs = np.vstack(imgs).astype(np.uint8)
        self.writer.add_image(tag, imgs, self.cur_iter, dataformats='HWC')

    def record_text(self, tag, text):
        self.writer.add_text(tag, text) 

    def printIterSummary(self, epoch, cur_iters, total_it, timer):
        msg = '{}\nIter: [{}]{:03d}/{:03d}\t\t'.format(
                timer.to_string(total_it - cur_iters), epoch, cur_iters, total_it)
        for k, v in self.iter_log[-1].items():
            msg += '{}: {:.6f}\t'.format(k, v) 
        print(msg + '\n')
        with open(self.txt_files[self.mode], 'a+') as f:
            f.write(msg + '\n')

    def close(self):
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, 'all_scalars.json'))
        self.writer.close()




