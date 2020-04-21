import torch


def mean_absolute_error(pred: torch.FloatTensor, y: torch.FloatTensor) -> float:
	return torch.mean((pred - y).abs()).item()


def mean_square_error(pred: torch.FloatTensor, y: torch.FloatTensor) -> float:
	return torch.mean((pred - y) ** 2).item()


class Metrics:
	def __init__(self):
		self.metrics = {
			'MSE': mean_square_error,
			'MAE': mean_absolute_error
		}
		self.total = {}.fromkeys(self.metrics.keys(), .0)
		self.batch = 0

	def update(self, pred, y):
		for k, metric in self.metrics.items():
			self.total[k] += metric(pred, y)
		self.batch += 1

	def clear(self):
		self.total = {}.fromkeys(self.metrics.keys(), .0)
		self.batch = 0

	@property
	def status(self):
		average = {k: v / self.batch for k, v in self.total.items()}
		average['RMSE'] = average['MSE'] ** .5
		return average
