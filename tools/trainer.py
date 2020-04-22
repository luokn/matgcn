import json
import os
from datetime import datetime
from functools import partial

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from tools.config import Config
from tools.metrics import Metrics
from tools.data import make_loaders
from tools.model import make_matgcn
from tools.utils import log_to_file, make_saved_dir, save_statistics, save_model, save_history


class Trainer:
	def __init__(self, conf: Config):
		self.conf = conf
		self.history = []
		self.requires_move = conf.device_for_model != conf.device_for_data
		self.saved_dir = make_saved_dir(conf.saved_dir)
		self.log_for_train = partial(log_to_file, f'{self.saved_dir}/train.log')
		self.log_for_validate = partial(log_to_file, f'{self.saved_dir}/validate.log')
		# load
		print('Loading data...')
		data_loaders, statistics = make_loaders(conf)
		self.data_for_train = data_loaders['train']
		self.data_for_validate = data_loaders['validate']
		# save_statistics(statistics, f'{self.saved_dir}/statistics.pth')
		# creat
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
			if epoch > int(0.2 * self.conf.epochs) and MAE < best:
				save_model(self.matgcn, f'{self.saved_dir}/model-{MAE:.2f}.pkl')
				best = MAE
		save_history(self.history, f'{self.saved_dir}/history.json')

	def train_epoch(self, epoch):
		total_loss, count = .0, 0
		with tqdm(total=len(self.data_for_train), desc='TRAIN', unit='batches') as bar:
			for b, (x, y) in enumerate(self.data_for_train):
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
				self.log_for_train(epoch=epoch, batch=b, loss=loss.item())
		return {'loss': total_loss / count}

	@torch.no_grad()
	def validate_epoch(self, epoch):
		metrics = Metrics()
		total_loss, count = .0, 0
		self.matgcn.eval()
		with tqdm(total=len(self.data_for_validate), desc='VALIDATE', unit='batches') as bar:
			for b, (x, y) in enumerate(self.data_for_validate):
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
				self.log_for_validate(epoch=epoch, batch=b, loss=loss.item(), **metrics.status)
		self.matgcn.train()
		return {'loss': total_loss / count, 'metrics': metrics.status}
