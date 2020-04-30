# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import torch


class Metrics:
	def __init__(self):
		self.count = 0
		self.AE, self.SE, = .0, .0
		self.MAE, self.RMSE = .0, .0

	def update(self, pred, y):
		self.count += pred.nelement()
		self.AE += torch.abs(pred - y).sum().item()
		self.SE += torch.pow(pred - y, 2).sum().item()
		self.MAE = self.AE / self.count
		self.RMSE = (self.SE / self.count) ** .5

	def clear(self):
		self.AE, self.SE, self.count = .0, .0, 0

	@property
	def state_dict(self):
		return {'MAE': self.MAE, 'RMSE': self.RMSE}
