import torch

from models.matgcn import MATGCN
from tools.utils import norm_adj_matrix


def matgcn(adj_file, n_nodes, out_timesteps, points_per_hour, device):
	mixin = dict(n_nodes=n_nodes, in_timesteps=points_per_hour, out_timesteps=out_timesteps,
				 A=norm_adj_matrix(adj_file, n_nodes, device))
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
				'tcn_dilations': [2, 4]
			}
		]
	}] * 5
	model = MATGCN(layers=layers, **mixin).to(device)
	for params in model.parameters(recurse=True):
		if params.dim() > 1:
			torch.nn.init.xavier_uniform_(params)
		else:
			torch.nn.init.uniform_(params)
	return model
