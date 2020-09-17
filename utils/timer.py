import time
import datetime
from collections import OrderedDict

class Timer():
    def __init__(self):
        self.reset_timer()
        self.start = time.time()
    
    def reset_timer(self):
        self.before = time.time() 
        self.timer = OrderedDict() 

    def update_time(self, key):
        self.timer[key] = time.time() - self.before
        self.before = time.time()

    def to_string(self, iters_left, short=False):
        iter_total = sum(self.timer.values())
        msg = "{:%Y-%m-%d %H:%M:%S}\tElapse: {}\tTimeLeft: {}\t".format(
                datetime.datetime.now(), 
                datetime.timedelta(seconds=round(time.time() - self.start)),
                datetime.timedelta(seconds=round(iter_total*iters_left))
                )
        if short:
            msg += '{}: {:.2f}s'.format('|'.join(self.timer.keys()), iter_total)
        else:
            msg += '\tIterTotal: {:.2f}s\t{}: {}  '.format(iter_total, 
                    '|'.join(self.timer.keys()), ' '.join('{:.2f}s'.format(x) for x in self.timer.values()))
        return msg

