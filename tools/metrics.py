import torch


class Metrics:
	def __init__(self):
		self.total = {'abs': .0, 'sqr': .0, }
		self.count = 0

	def update(self, pred: torch.FloatTensor, y: torch.FloatTensor):
		self.total['abs'] += torch.sum((pred - y).abs()).item()
		self.total['sqr'] += torch.sum((pred - y) ** 2).item()
		self.count += pred.nelement()

	def clear(self):
		self.total = {'abs': .0, 'sqr': .0, }
		self.count = 0

	@property
	def status(self):
		return {
			'MAE': self.total['abs'] / self.count,
			'RMSE': (self.total['sqr'] / self.count) ** .5
		}
