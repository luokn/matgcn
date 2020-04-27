# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import json
from functools import partial

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from tools.config import Config
from tools.data import load_data
from tools.metrics import Metrics
from tools.model import matgcn
from tools.utils import log_to_file, make_saved_dir


class Trainer:
	def __init__(self, conf: Config):
		self.device, self.epochs = conf.device_for_model, conf.epochs
		self.requires_move = conf.device_for_model != conf.device_for_data
		self.saved_dir = make_saved_dir(conf.saved_dir)
		self.train_log = partial(log_to_file, f'{self.saved_dir}/train.log')
		self.validate_log = partial(log_to_file, f'{self.saved_dir}/validate.log')
		print('Loading...')
		# load
		loaders, statistics = load_data(conf.data_file, conf.batch_size, conf.data_split,
										conf.points_per_hour, conf.device_for_data)
		self.train_loader, self.validate_loader = loaders
		torch.save(statistics, f'{self.saved_dir}/statistics.pth')
		# create
		self.model = matgcn(conf.adj_file, conf.n_nodes, conf.out_timesteps,
							conf.points_per_hour, conf.device_for_model)
		self.optimizer = Adam(self.model.parameters(), lr=conf.lr)
		self.criterion = MSELoss().to(conf.device_for_model)

	def fit(self):
		print('Training...')
		# train
		best = float('inf')
		history = []
		for epoch in range(self.epochs):
			print(f"Epoch: {epoch + 1}")
			history.append({
				'train': self.train_epoch(epoch),
				'validate': self.validate_epoch(epoch)
			})
			MAE = history[-1]['validate']['metrics']['MAE']
			if epoch >= int(.2 * self.epochs) and MAE < best:
				torch.save(self.model.state_dict(), f'{self.saved_dir}/model-{MAE:.2f}.pkl')
				best = MAE
		open(f'{self.saved_dir}/history.json', 'w').write(json.dumps(history))

	def train_epoch(self, epoch):
		total_loss, count = .0, 0
		with tqdm(total=len(self.train_loader), desc='Train', unit='batches') as bar:
			for idx, batch in enumerate(self.train_loader):
				x, h, d, y = [it.to(self.device) for it in batch] if self.requires_move else batch
				self.optimizer.zero_grad()
				pred = self.model(x, h, d)
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
				self.train_log(epoch=epoch, batch=idx, loss=loss.item())
		return {'loss': total_loss / count}

	@torch.no_grad()
	def validate_epoch(self, epoch):
		metrics = Metrics()
		total_loss, count = .0, 0
		self.model.eval()
		with tqdm(total=len(self.validate_loader), desc='Validate', unit='batches') as bar:
			for idx, batch in enumerate(self.validate_loader):
				x, h, d, y = [it.to(self.device) for it in batch] if self.requires_move else batch
				pred = self.model(x, h, d)
				loss = self.criterion(pred, y)
				# update statistics
				metrics.update(pred, y)
				count += len(pred)
				total_loss += loss.item()
				# update progress bar
				bar.update()
				bar.set_postfix(**{
					k: f'{v:.2f}' for k, v in metrics.state_dict.items()
				}, loss=f'{total_loss / count:.2f}')
				self.validate_log(epoch=epoch, batch=idx, loss=loss.item(), **metrics.state_dict)
		self.model.train()
		return {'loss': total_loss / count, 'metrics': metrics.state_dict}
