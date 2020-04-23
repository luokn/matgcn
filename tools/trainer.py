import json
from functools import partial

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from tools.config import Config
from tools.metrics import Metrics
from tools.model import make_matgcn
from tools.data import make_loaders
from tools.utils import log_to_file, make_saved_dir


class Trainer:
	def __init__(self, conf: Config):
		self.conf = conf
		self.history = []
		self.requires_move = conf.device_for_model != conf.device_for_data
		self.saved_dir = make_saved_dir(conf.saved_dir)
		self.train_log = partial(log_to_file, f'{self.saved_dir}/train.log')
		self.validate_log = partial(log_to_file, f'{self.saved_dir}/validate.log')
		# load
		print('Loading data...')
		data_loaders, statistics = make_loaders(conf)
		self.train_loader = data_loaders['train']
		self.validate_loader = data_loaders['validate']
		# torch.save(statistics, f'{self.saved_dir}/statistics.pth')
		# create
		print('Creating model...')
		self.matgcn = make_matgcn(conf)
		self.optimizer = Adam(self.matgcn.parameters(), lr=conf.lr)
		self.criterion = MSELoss().to(conf.device_for_model)

	def run(self):
		# train
		print('Training...')
		best = float('inf')
		self.history.clear()
		for epoch in range(self.conf.epochs):
			print(f"EPOCH: {epoch + 1}")
			self.history.append({
				'train': self.train_epoch(epoch),
				'validate': self.validate_epoch(epoch)
			})
			MAE = self.history[-1]['validate']['metrics']['MAE']
			if epoch >= int(.2 * self.conf.epochs) and MAE < best:
				torch.save(self.matgcn, f'{self.saved_dir}/model-{MAE:.2f}.pkl')
				best = MAE
		open(f'{self.saved_dir}/history.json', 'w').write(json.dumps(self.history))

	def train_epoch(self, epoch):
		total_loss, count = .0, 0
		with tqdm(total=len(self.train_loader), desc='TRAIN', unit='batches') as bar:
			for b, (x, y) in enumerate(self.train_loader):
				if self.requires_move:
					x, y = x.to(self.conf.device_for_model), y.to(self.conf.device_for_model)
				self.optimizer.zero_grad()
				pred = self.matgcn(x)
				loss = self.criterion(pred, y)
				loss.backward()
				self.optimizer.step()
				# update statistics
				count += len(pred)
				total_loss += loss.item()
				# update progress bar
				bar.update()
				bar.set_postfix(loss=f'{total_loss / count:.2f}')
				# log to file
				self.train_log(epoch=epoch, batch=b, loss=loss.item())
		return {'loss': total_loss / count}

	@torch.no_grad()
	def validate_epoch(self, epoch):
		metrics = Metrics()
		total_loss, count = .0, 0
		self.matgcn.eval()
		with tqdm(total=len(self.validate_loader), desc='VALIDATE', unit='batches') as bar:
			for b, (x, y) in enumerate(self.validate_loader):
				if self.requires_move:
					x, y = x.to(self.conf.device_for_model), y.to(self.conf.device_for_model)
				pred = self.matgcn(x)
				loss = self.criterion(pred, y)
				# update statistics
				metrics.update(pred, y)
				count += len(pred)
				total_loss += loss.item()
				# update progress bar
				bar.update()
				bar.set_postfix(**{
					k: f'{v:.2f}' for k, v in metrics.status.items()
				}, loss=f'{total_loss / count:.2f}')
				self.validate_log(epoch=epoch, batch=b, loss=loss.item(), **metrics.status)
		self.matgcn.train()
		return {'loss': total_loss / count, 'metrics': metrics.status}
