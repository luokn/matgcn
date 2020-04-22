import json
import os
from datetime import datetime

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from tools.config import Config
from tools.metrics import Metrics
from tools.data import make_loaders
from tools.model import make_matgcn


class Trainer:
	def __init__(self, conf: Config):
		self.conf = conf
		self.history = []
		self.requires_move = conf.device_for_model != conf.device_for_data
		# load
		print('Loading data...')
		data_loaders, self.statistics = make_loaders(conf)
		self.data_for_train = data_loaders['train']
		self.data_for_validate = data_loaders['validate']
		# creat
		print('Creating model...')
		self.matgcn = make_matgcn(conf)
		self.optimizer = Adam(self.matgcn.parameters(), lr=conf.lr)
		self.criterion = MSELoss().to(conf.device_for_model)

	def run(self):
		# train
		print('Training...')
		self.history.clear()
		for epoch in range(self.conf.epochs):
			print(f"EPOCH: {epoch + 1}")
			self.history.append({
				'train': self.train_epoch(),
				'validate': self.validate_epoch()
			})
		self.save()

	def train_epoch(self):
		total_loss, count = .0, 0
		with tqdm(total=len(self.data_for_train), desc='TRAIN', unit='batches') as bar:
			for x, y in self.data_for_train:
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
		return {'loss': total_loss / count}

	@torch.no_grad()
	def validate_epoch(self):
		metrics = Metrics()
		total_loss, count = .0, 0
		self.matgcn.eval()
		with tqdm(total=len(self.data_for_validate), desc='VALIDATE', unit='batches') as bar:
			for x, y in self.data_for_validate:
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
		self.matgcn.train()
		return {'loss': total_loss / count, 'metrics': metrics.status}

	def save(self, history=True, statistics=False, model=False):
		if not os.path.exists(self.conf.saved_dir):
			os.mkdir(self.conf.saved_dir)
		saved_dir = os.path.join(self.conf.saved_dir, datetime.now().strftime('%m-%d-%H-%M-%S'))
		if not os.path.exists(saved_dir):
			os.mkdir(saved_dir)
		if history:
			# save history
			with open(f'{saved_dir}/history.json', 'w') as f:
				f.write(json.dumps(self.history))
		if statistics:
			# save normalizers
			torch.save(self.statistics, f'{saved_dir}/statistics.pth')
		if model:
			# save model
			torch.save(self.matgcn, f'{saved_dir}/model.pkl')
