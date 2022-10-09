# author: lgx
# date: 2022/7/7 22:04

import time
from collections import defaultdict


class Recorder:
    def __init__(self, work_dir=None, save_log=False):
        self.cur_time = time.time()
        self.save_log = save_log
        self.log_path = '{}/log.txt'.format(work_dir) if save_log else None
        self.timer = defaultdict(float)

    def print_time(self):
        self.print_log("Current time:  " + time.asctime())

    def print_log(self, print_str, print_time=True):
        if print_time:
            print_str = "[ " + time.asctime() + ' ] ' + print_str
        print(print_str)

        if self.save_log:
            with open(self.log_path, 'a') as f:
                f.write(print_str + '\n')

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.cur_time = time.time()
        return split_time

    def timer_reset(self):
        self.cur_time = time.time()
        self.timer = defaultdict(float)

    def record_timer(self, key):
        self.timer[key] += self.split_time()

    def print_time_statistics(self, show_proportion=True):
        if show_proportion:
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(self.timer.values()))))
                for k, v in self.timer.items()}
        else:
            proportion = {
                k: '{:02d}ms'.format(int(round(v*1000)))
                for k, v in self.timer.items()}

        output = '\tTime consumption:'
        for k, v in proportion.items():
            output += ' [{}]{}'.format(k, v)

        self.print_log(output)




