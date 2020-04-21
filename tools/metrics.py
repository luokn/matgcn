import torch


class Metrics:
	def __init__(self):
		self.total = {}.fromkeys(['abs', 'sqr'], .0)
		self.count = 0

	def update(self, pred, y):
		self.total['abs'] += torch.abs(pred - y).sum().item()
		self.total['sqr'] += torch.pow(pred - y, 2).sum().item()
		self.count += pred.nelement()

	def clear(self):
		self.total = {}.fromkeys(['abs', 'sqr'], .0)
		self.count = 0

	@property
	def status(self):
		return {
			'MAE': self.total['abs'] / self.count,
			'RMSE': (self.total['sqr'] / self.count) ** .5
		}
