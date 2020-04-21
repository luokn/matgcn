import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.config import Config


def generate_sequences(data, tp):
	n_timesteps, n_vertices, n_channels = data.shape
	n_sequences = n_timesteps - tp - 168 * tp
	# generate
	ranges = [(- i * tp, -i * tp + tp) for i in [1, 2, 3, 4, 24, 24 * 7]]
	Y = torch.zeros(n_sequences, n_vertices, tp, device=data.device)
	X = torch.zeros(n_sequences, len(ranges), n_channels, n_vertices, tp, device=data.device)
	for i, t in enumerate(range(168 * tp, n_timesteps - tp)):
		torch.stack([
			data[t + start:t + end].transpose(0, 2) for start, end in ranges
		], dim=0, out=X[i])
		Y[i] = data[t:t + tp, :, 0].T
	return X, Y


def normalize_sequences(x, split):
	std = torch.std(x[:split], dim=0, keepdim=True)
	mean = torch.mean(x[:split], dim=0, keepdim=True)
	x -= mean
	x /= std
	return dict(std=std, mean=mean)


def make_loaders(conf: Config):
	data = torch.from_numpy(numpy.load(conf.data_file)['data']).to(conf.device_for_data)
	X, Y = generate_sequences(data.float(), conf.points_per_hour)
	split = int(len(Y) * conf.data_split)
	statistics = normalize_sequences(X, split)
	dataset1 = TensorDataset(X[:split], Y[:split])  # for train
	dataset2 = TensorDataset(X[split:], Y[split:])  # for validate
	data_loaders = {
		"train": DataLoader(dataset1, batch_size=conf.batch_size, shuffle=True),
		"validate": DataLoader(dataset2, batch_size=conf.batch_size, shuffle=True)
	}
	return data_loaders, statistics
