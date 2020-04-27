# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import torch


class Metrics:
	def __init__(self):
		self.AE, self.SE, self.count = .0, .0, 0

	def update(self, pred, y):
		self.AE += torch.abs(pred - y).sum().item()
		self.SE += torch.pow(pred - y, 2).sum().item()
		self.count += pred.nelement()

	def clear(self):
		self.AE, self.SE, self.count = .0, .0, 0

	@property
	def state_dict(self):
		return {
			'MAE': self.AE / self.count,
			'RMSE': (self.SE / self.count) ** .5
		}
