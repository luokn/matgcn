# -*- coding: utf-8 -*-
# @Date  : 2020/4/30
# @Author: Luokun
# @Email : olooook@outlook.com
from time import time

from datetime import timedelta


class ProgressBar:
    def __init__(self, total: int, n_circles=50):
        self.count = 0
        self.total = total
        self.n_circles = n_circles
        self.start_time = time()
        self.last_line_len = -1

    def update(self, postfix='', n=1):
        self.count += n
        n_filled = self.n_circles * self.count // self.total
        time_str = timedelta(seconds=int(time() - self.start_time))
        if n_filled < self.n_circles:
            circles = n_filled * '●' + (self.n_circles - n_filled) * '○'
            line = f'\r[{circles}] [{self.count}/{self.total} {time_str}] {postfix}'
        else:
            line = f'\r[{self.count}/{self.total} {time_str}] {postfix}'
        if self.last_line_len > len(line):
            line = line.ljust(self.last_line_len)
        self.last_line_len = len(line)
        print(line, end='')

    def close(self):
        self.count, self.last_line_len = 0, -1
        print('')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
