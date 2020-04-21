import math

import torch
from torch import FloatTensor
from torch.nn import Conv2d, LayerNorm, Module, Parameter, Sequential, ModuleList, ReLU, Dropout
from torch.nn.utils import weight_norm


class Attention(Module):
	def __init__(self, dk, requires_value=False):
		super(Attention, self).__init__()
		self.sqrt_dk = math.sqrt(dk)
		self.requires_value = requires_value
		self.W1 = Parameter(torch.zeros(dk, 10), requires_grad=True)
		self.W2 = Parameter(torch.zeros(10, dk), requires_grad=True)

	def forward(self, x: FloatTensor):
		x_out = x.reshape(*x.shape[:2], -1)
		# [B * A * Dk] @ [Dk * Dk] @ [B * Dk * A] => [B * A * A]
		att = x_out @ self.W1 @ self.W2 @ x_out.transpose(1, 2)
		att = torch.softmax(att / self.sqrt_dk, dim=-1)
		return (att @ x_out).reshape_as(x) if self.requires_value else att


class GCN(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, A):
		super(GCN, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(in_channels, out_channels), requires_grad=True)
		self.att = Attention(in_channels * in_timesteps, requires_value=False)

	def forward(self, x: FloatTensor):
		# In : B * C_i * V * T
		# Out: B * C_o * V * T
		att = self.att(x.transpose(1, 2))  # => [B * V * V]
		# [B * V * V] @ [T * B * V * C_i] @ [C_i * C_o] => [T * B * V * C_o]
		x_out = (att * self.A) @ x.permute(3, 0, 2, 1) @ self.W
		return x_out.permute(1, 3, 2, 0)


class TCN(Module):
	def __init__(self, in_channels, n_vertices, dilations):
		super(TCN, self).__init__()
		self.att = Attention(n_vertices * in_channels, requires_value=True)
		layers = []
		for dilation in dilations:
			layers += [
				weight_norm(Conv2d(in_channels, in_channels, [1, 3], padding=[0, dilation], dilation=[1, dilation])),
				ReLU(),
				Dropout(.2),
				weight_norm(Conv2d(in_channels, in_channels, [1, 3], padding=[0, dilation], dilation=[1, dilation])),
				ReLU(),
				Dropout(.2)
			]
		self.layers = Sequential(*layers)

	def forward(self, x: FloatTensor):
		# In : B * C * V * T
		# Out: B * C * V * T
		x_out = self.att(x.transpose(1, 3))
		x_out = self.layers(x_out.transpose(1, 3))
		return x_out


class MATGCNBlock(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, n_vertices, tcn_dilations, **kwargs):
		super(MATGCNBlock, self).__init__()
		self.att = Attention(n_vertices * in_timesteps, requires_value=True)
		self.res = Conv2d(in_channels, out_channels, kernel_size=1)
		self.gcn = GCN(in_channels, out_channels, in_timesteps, kwargs['A'])
		self.tcn = TCN(out_channels, n_vertices, tcn_dilations)
		self.ln = LayerNorm(normalized_shape=out_channels)

	def forward(self, x: FloatTensor):
		# In : B * V * C_i * T_i
		# Out: B * V * C_o * T_o
		x_out = self.att(x)
		x_out = self.gcn(x_out)
		x_out = self.tcn(x_out)
		x_out += self.res(x)
		x_out = x_out.relu_().transpose(1, 3)
		return self.ln(x_out).transpose(1, 3)


class MATGCNLayer(Module):
	def __init__(self, blocks, **kwargs):
		super(MATGCNLayer, self).__init__()
		self.blocks = Sequential(*[MATGCNBlock(**block, **kwargs) for block in blocks])
		self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'],
						 kernel_size=[1, blocks[-1]['out_channels']])

	def forward(self, x: FloatTensor):
		# In : B * C_i * V * T_i
		# Out: B * C_o * V * T_o
		x = self.blocks(x)
		x = self.fc(x.transpose(1, 3))  # => (B * T_o * V * 1)
		return x[..., 0].transpose(1, 2)


class MATGCN(Module):
	def __init__(self, layers, **kwargs):
		super(MATGCN, self).__init__()
		n_vertices, out_timesteps = kwargs['n_vertices'], kwargs['out_timesteps']
		self.layers = ModuleList([MATGCNLayer(**layer, **kwargs) for layer in layers])
		self.W = Parameter(torch.zeros(len(layers), n_vertices, out_timesteps), requires_grad=True)

	# self.att_in = Attention(n_vertices * out_timesteps, requires_value=True)
	# self.att_out = Attention(n_vertices * out_timesteps, requires_value=True)

	def forward(self, X: FloatTensor):
		# 混合通道注意力
		return sum(map(lambda layer, x, w: layer(x) * w, self.layers, X.unbind(1), self.W.unbind(0)))
