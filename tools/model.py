import torch

from tools.config import Config
from models.matgcn import MATGCN
from tools.utils import norm_adj_matrix


def make_matgcn(conf: Config):
	mixin = dict(
		n_vertices=conf.n_vertices,
		in_timesteps=conf.points_per_hour,
		out_timesteps=conf.out_timesteps,
		A=norm_adj_matrix(conf.adj_file, conf.n_vertices, conf.device_for_model),
	)
	layers = [{
		"blocks": [
			{
				'in_channels': 3,
				'out_channels': 64,
				'tcn_dilations': [1, 2]
			},
			{
				'in_channels': 64,
				'out_channels': 64,
				'tcn_dilations': [1, 2]
			}
		]
	}] * 5
	model = MATGCN(layers=layers, **mixin).to(conf.device_for_model)
	for params in model.parameters(recurse=True):
		if params.dim() > 1:
			torch.nn.init.xavier_uniform_(params)
		else:
			torch.nn.init.uniform_(params)
	return model
