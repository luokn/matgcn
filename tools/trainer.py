# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import json
from functools import partial

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from tools.config import Config
from tools.data import load_data
from tools.metrics import Metrics
from tools.model import matgcn
from tools.progress import ProgressBar
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
			train_loss = self.train_epoch(epoch)
			validate_loss, metrics = self.validate_epoch(epoch)
			if epoch >= int(.2 * self.epochs) and metrics.MAE < best:
				best = metrics.MAE
				torch.save(self.model.state_dict(), f'{self.saved_dir}/model-{best:.2f}.pkl')
			history.append(dict(train_loss=train_loss, validate_loss=validate_loss, metrics=metrics.state_dict))
		open(f'{self.saved_dir}/history.json', 'w').write(json.dumps(history))

	def train_batch(self, batch):
		x, h, d, y = [it.to(self.device) for it in batch] if self.requires_move else batch
		self.optimizer.zero_grad()
		pred = self.model(x, h, d)
		loss = self.criterion(pred, y)
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def validate_batch(self, batch, metrics):
		x, h, d, y = [it.to(self.device) for it in batch] if self.requires_move else batch
		pred = self.model(x, h, d)
		loss = self.criterion(pred, y)
		metrics.update(pred, y)
		return loss.item()

	def train_epoch(self, epoch):
		total_loss, average_loss = .0, .0
		with ProgressBar(total=len(self.train_loader)) as bar:
			for idx, batch in enumerate(self.train_loader):
				loss = self.train_batch(batch)
				total_loss += loss
				average_loss = total_loss / (idx + 1)
				bar.update(postfix=f'[Train] loss={average_loss:.2f}')
				self.train_log(epoch=epoch, batch=idx, loss=loss)
		return average_loss

	@torch.no_grad()
	def validate_epoch(self, epoch):
		metrics = Metrics()
		self.model.eval()
		total_loss, average_loss = .0, .0
		with ProgressBar(total=len(self.validate_loader)) as bar:
			for idx, batch in enumerate(self.validate_loader):
				loss = self.validate_batch(batch, metrics)
				total_loss += loss
				average_loss = total_loss / (idx + 1)
				bar.update(
					postfix=f'[Validate] loss={average_loss:.2f} MAE={metrics.MAE:.2f} RMSE={metrics.RMSE:.2f}'
				)
				self.validate_log(epoch=epoch, batch=idx, loss=loss, MAE=metrics.MAE, RMSE=metrics.RMSE)
		self.model.train()
		return average_loss, metrics
