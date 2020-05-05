# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import torch
from torch import FloatTensor, LongTensor
from torch.nn import Conv2d, LayerNorm, Module, Parameter, Sequential, ModuleList, Embedding, BatchNorm2d


class CAttention(Module):
	def __init__(self, n_nodes, n_timesteps):
		super(CAttention, self).__init__()
		self.W = Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
		self.alpha = Parameter(torch.zeros(n_nodes), requires_grad=True)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, C, N, T]
		"""
		K = Q = x.transpose(2, 3) @ self.alpha  # [B, C, T]
		A = torch.softmax(K @ self.W @ Q.transpose(1, 2), dim=-1)  # [B, C, C]
		return (A @ x.reshape(*x.shape[:2], -1)).view_as(x)  # [B, C, N, T]


class SAttention(Module):
	def __init__(self, n_channels, n_timesteps, A):
		super(SAttention, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
		self.alpha = Parameter(torch.zeros(n_channels), requires_grad=True)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, N, N]
		"""
		K = Q = x.permute(0, 2, 3, 1) @ self.alpha  # [B, N, T]
		A = torch.softmax(K @ self.W @ Q.transpose(1, 2), dim=-1)  # [B, N, N]
		return self.A * A  # [B, N, N]


class TAttention(Module):
	def __init__(self, n_channels, n_nodes):
		super(TAttention, self).__init__()
		self.W1 = Parameter(torch.zeros(10, n_nodes), requires_grad=True)
		self.W2 = Parameter(torch.zeros(10, n_nodes), requires_grad=True)
		self.alpha = Parameter(torch.zeros(n_channels), requires_grad=True)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, C, N, T]
		"""
		x = x.transpose(1, 3)  # [B, T, N, C]
		K = Q = x @ self.alpha  # [B, T, N]
		A = torch.softmax((K @ self.W1.T) @ (Q @ self.W2.T).transpose(1, 2), dim=-1)  # [B, T, T]
		return (A @ x.reshape(*x.shape[:2], -1)).view_as(x).transpose(1, 3)  # [B, C, N, T]


class GCNBlock(Module):
	def __init__(self, in_channels, out_channels, n_timesteps, A):
		super(GCNBlock, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)  # [C_o, C_i]
		self.att = SAttention(n_channels=in_channels, n_timesteps=n_timesteps, A=A)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T]
		:return: [B, C_o, N, T]
		"""
		A = self.att(x)  # [B, N, N]
		# [B, N, N] @ [T, B, N, C_i] @ [C_i, C_o]
		x_out = A @ x.permute(3, 0, 2, 1) @ self.W.T  # [T, B, N, C_o]
		return x_out.permute(1, 3, 2, 0)  # [B, C_o, N, T]


class TCNBlock(Module):
	def __init__(self, in_channels, n_nodes, dilations):
		super(TCNBlock, self).__init__()
		self.att = TAttention(n_channels=in_channels, n_nodes=n_nodes)
		self.convs = ModuleList([
			Conv2d(in_channels, in_channels, [1, 2], padding=[0, dilation], dilation=[1, dilation])
			for dilation in dilations
		])

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, C, N, T]
		"""
		x_out = self.att(x)  # [B, C, N, T]
		for conv in self.convs:
			x_out = conv(x_out)  # [B, C, N, T + P]
			x_out = torch.relu(x_out[..., :-conv.padding[1]])  # [B, C, N, T]
		return x_out  # [B, C, N, T]


class MATGCNBlock(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, tcn_dilations, n_nodes, **kwargs):
		super(MATGCNBlock, self).__init__()
		self.seq = Sequential(
			LayerNorm([n_nodes, in_timesteps]),
			CAttention(n_nodes, in_timesteps),
			GCNBlock(in_channels, out_channels, in_timesteps, kwargs['A']),
			TCNBlock(out_channels, n_nodes, tcn_dilations),
		)
		self.res = Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T]
		:return: [B, C_o, N, T]
		"""
		x_out = self.seq(x) + self.res(x)  # [B, C_o, N, T]
		return x_out.relu_()


class MATGCNLayer(Module):
	def __init__(self, blocks, **kwargs):
		super(MATGCNLayer, self).__init__()
		self.blocks = Sequential(*[MATGCNBlock(**block, **kwargs) for block in blocks])
		self.ln = LayerNorm(blocks[-1]['out_channels'])
		self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'], [1, blocks[-1]['out_channels']])

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T_i]
		:return: [B, C_o, N, T_o]
		"""
		x = self.blocks(x)  # [B, C_o, N, T_o]
		x = self.ln(x.transpose(1, 3))
		x = self.fc(x)  # [B, T_o, N, 1]
		return x[..., 0].transpose(1, 2)  # [B, N, T_o]


class MATGCN(Module):
	def __init__(self, layers, **kwargs):
		super(MATGCN, self).__init__()
		self.n_nodes, out_timesteps = kwargs['n_nodes'], kwargs['out_timesteps']
		self.layers = ModuleList([MATGCNLayer(**layer, **kwargs) for layer in layers])
		self.d_embed = Embedding(7, len(layers) * self.n_nodes * out_timesteps)
		self.h_embed = Embedding(24, len(layers) * self.n_nodes * out_timesteps)

	def forward(self, X: FloatTensor, H: LongTensor, D: LongTensor):
		"""
		:param X: [B, L, C, N, T_i]
		:param H: [B]
		:param D: [B]
		:return: [B, N, T_o]
		"""
		G = self.h_embed(H) + self.d_embed(D)  # [(B * L * N * T_o)]
		G = G.view(len(G), len(self.layers), self.n_nodes, -1)  # [B * L * N * T_o]
		return sum(map(lambda layer, x, gate: layer(x) * gate, self.layers, X.unbind(1), G.unbind(1)))
