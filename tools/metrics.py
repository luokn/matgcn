import torch


class Metrics:
	def __init__(self):
		self.total = {}.fromkeys(['AE', 'SE'], .0)
		self.count = 0

	def update(self, pred, y):
		self.total['AE'] += torch.abs(pred - y).sum().item()
		self.total['SE'] += torch.pow(pred - y, 2).sum().item()
		self.count += pred.nelement()

	def clear(self):
		self.total = {}.fromkeys(['AE', 'SE'], .0)
		self.count = 0

	@property
	def status(self):
		return {
			'MAE': self.total['AE'] / self.count,
			'RMSE': (self.total['SE'] / self.count) ** .5
		}
