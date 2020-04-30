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
		self.total_str_len = len(str(total))

	def update(self, postfix='', n=1):
		self.count += n
		circles = self.n_circles * self.count // self.total
		tm_delta = timedelta(seconds=int(time() - self.start_time))
		if circles < self.n_circles:
			progress = circles * '●' + (self.n_circles - circles) * '○'
			line = f'\r[{progress}] [{self.count}/{self.total} {tm_delta}] {postfix}'
		else:
			line = f'\r[{self.count}/{self.total} {tm_delta}] {postfix}'
		if self.last_line_len > len(line):
			line = line.ljust(self.last_line_len)
		print(line, end='')
		self.last_line_len = len(line)

	def close(self):
		self.count = 0
		self.last_line_len = -1
		print('')

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()
