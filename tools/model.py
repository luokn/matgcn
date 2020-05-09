from torch.nn.init import xavier_normal_, uniform_

from models.matgcn import MATGCN
from tools.utils import read_adj_matrix


def matgcn(adj_file, n_nodes, out_timesteps, points_per_hour, device):
    mixin = dict(A=read_adj_matrix(adj_file, n_nodes, device),
                 n_nodes=n_nodes,
                 in_timesteps=points_per_hour,
                 out_timesteps=out_timesteps)
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
    for param in model.parameters():
        if param.ndim >= 2:
            xavier_normal_(param)
        else:
            uniform_(param)
    return model
