# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import math

import torch
from torch import FloatTensor, LongTensor
from torch.nn import Conv2d, LayerNorm, Module, Parameter, Sequential, ModuleList, Embedding


class Attention(Module):
	def __init__(self, dk, requires_value=False):
		super(Attention, self).__init__()
		self.sqrt_dk = math.sqrt(dk)
		self.requires_value = requires_value
		self.W1 = Parameter(torch.zeros(dk, 10), requires_grad=True)
		self.W2 = Parameter(torch.zeros(10, dk), requires_grad=True)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, A, ...]
		:return: [B, A, ...] or [B, A, A]
		"""
		x_out = x.reshape(*x.shape[:2], -1)  # => [B, A, D_k]
		A = x_out @ self.W1 @ self.W2 @ x_out.transpose(1, 2)  # => [B, A, A]
		A = torch.softmax(A / self.sqrt_dk, dim=-1)  # => [B, A, A]
		return (A @ x_out).reshape_as(x) if self.requires_value else A  # => [B, A, ...] or [B, A, A]


class GCNBlock(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, A):
		super(GCNBlock, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(in_channels, out_channels), requires_grad=True)  # [C_i, C_o]
		self.att = Attention(in_channels * in_timesteps, requires_value=False)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T]
		:return: [B, C_o, N, T]
		"""
		A = self.att(x.transpose(1, 2)) * self.A  # => [B, N, N]
		# [B, N, N] @ [T, B, N, C_i] @ [C_i, C_o]
		x_out = A @ x.permute(3, 0, 2, 1) @ self.W  # => [T, B, N, C_o]
		return x_out.permute(1, 3, 2, 0)  # => [B, C_o, N, T]


class TCNBlock(Module):
	def __init__(self, in_channels, n_nodes, dilations):
		super(TCNBlock, self).__init__()
		self.att = Attention(n_nodes * in_channels, requires_value=True)
		self.dilations = dilations
		self.convs = ModuleList([
			Conv2d(in_channels, in_channels, [1, 2], padding=[0, dilation], dilation=[1, dilation])
			for dilation in dilations
		])

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, C, N, T]
		"""
		x_out = self.att(x.transpose(1, 3)).transpose(1, 3)  # => [B, C, N, T]
		for conv, dilation in zip(self.convs, self.dilations):
			x_out = conv(x_out)  # => [B, C, N, T + D]
			x_out = torch.relu(x_out[..., :-dilation])  # => [B, C, N, T]
		return x_out  # => [B, C, N, T]


class MATGCNBlock(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, n_nodes, tcn_dilations, **kwargs):
		super(MATGCNBlock, self).__init__()
		self.seq = Sequential(
			Attention(n_nodes * in_timesteps, requires_value=True),
			GCNBlock(in_channels, out_channels, in_timesteps, kwargs['A']),
			TCNBlock(out_channels, n_nodes, tcn_dilations),
		)
		self.res = Conv2d(in_channels, out_channels, kernel_size=1)
		self.ln = LayerNorm(normalized_shape=out_channels)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T]
		:return: [B, C_o, N, T]
		"""
		x_out = self.seq(x) + self.res(x)  # => [B, C_o, N, T]
		x_out = x_out.relu_().transpose(1, 3)  # => [B, T, N, T_o]
		return self.ln(x_out).transpose(1, 3)  # => [B, C_o, N, T]


class MATGCNLayer(Module):
	def __init__(self, blocks, **kwargs):
		super(MATGCNLayer, self).__init__()
		self.blocks = Sequential(*[MATGCNBlock(**block, **kwargs) for block in blocks])
		self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'], [1, blocks[-1]['out_channels']])

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T_i]
		:return: [B, C_o, N, T_o]
		"""
		x = self.blocks(x)  # => [B, C_o, N, T_o]
		x = self.fc(x.transpose(1, 3))  # => [B, T_o, N, 1]
		return x[..., 0].transpose(1, 2)  # => [B, N, T_o]


class MATGCN(Module):
	def __init__(self, layers, **kwargs):
		super(MATGCN, self).__init__()
		self.n_nodes = kwargs['n_nodes']
		self.layers = ModuleList([MATGCNLayer(**layer, **kwargs) for layer in layers])
		self.h_embed = Embedding(24, len(layers) * self.n_nodes * kwargs['out_timesteps'])
		self.d_embed = Embedding(7, len(layers) * self.n_nodes * kwargs['out_timesteps'])

	def forward(self, X: FloatTensor, H: LongTensor, D: LongTensor):
		"""
		:param X: [B, L, C, N, T_i]
		:param H: [B]
		:param D: [B]
		:return: [B, N, T_o]
		"""
		G = self.h_embed(H) + self.d_embed(D)  # => [(B * L * N * T_o)]
		G = G.view(len(G), len(self.layers), self.n_nodes, -1)  # => [B * L * N * T_o]
		return sum(map(lambda layer, x, gate: layer(x) * gate, self.layers, X.unbind(1), G.unbind(1)))
