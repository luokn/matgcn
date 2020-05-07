# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import torch
from torch import FloatTensor, LongTensor
from torch.nn import Conv2d, LayerNorm, Module, Parameter, Sequential, ModuleList, Embedding


class SAttention(Module):
	def __init__(self, n_channels, n_timesteps):
		super(SAttention, self).__init__()
		self.W = Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
		self.alpha = Parameter(torch.zeros(n_channels), requires_grad=True)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, N, N]
		"""
		# k_{n,t} = q_{n,t} = x_{i,n,t} \alpha^{i}
		K = Q = torch.einsum('bint,i->bnt', x, self.alpha)  # [B, N, T]
		# [B, N, T] @ [T, T] @ [B, T, N]
		return torch.softmax(K @ self.W @ Q.transpose(1, 2), dim=-1)  # [B, N, N]


class GCNBlock(Module):
	def __init__(self, in_channels, out_channels, n_timesteps, A):
		super(GCNBlock, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)  # [C_o, C_i]
		self.s_att = SAttention(n_channels=in_channels, n_timesteps=n_timesteps)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T]
		:return: [B, C_o, N, T]
		"""
		A = self.s_att(x) * self.A  # [B, N, N]
		# [B, N, N] @ [T, B, N, C_i] @ [C_i, C_o]
		x_out = A @ x.permute(3, 0, 2, 1) @ self.W.T  # [T, B, N, C_o]
		return x_out.permute(1, 3, 2, 0)  # [B, C_o, N, T]


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
		# k_{t,n} = q_{t,n} = x_{i,n,t} \alpha^{i}
		K = Q = torch.einsum('bint,i->btn', x, self.alpha)  # [B, T, N]
		A = torch.softmax((K @ self.W1.T) @ (Q @ self.W2.T).transpose(1, 2), dim=-1)  # [B, T, T]
		# y_{c,n,t} = a_{t,i} x_{c,n,i}
		return torch.einsum('bti,bcni->bcnt', A, x)  # [B, C, N, T]


class Chomp(Module):
	def __init__(self, chomp_size):
		super(Chomp, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x: FloatTensor):
		return x[..., :-self.chomp_size]


class TCNBlock(Module):
	def __init__(self, in_channels, n_nodes, dilations):
		super(TCNBlock, self).__init__()
		seq = [
			TAttention(n_channels=in_channels, n_nodes=n_nodes)
		]
		for dilation in dilations:
			seq += [
				Conv2d(in_channels, in_channels, [1, 2], padding=[0, dilation], dilation=[1, dilation]),
				Chomp(dilation)
			]
		self.seq = Sequential(*seq)

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C, N, T]
		:return: [B, C, N, T]
		"""
		return self.seq(x)  # [B, C, N, T]


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
		# k_{c,t} = q_{c,t} = x_{c,i,t} \alpha^{i}
		K = Q = torch.einsum('bcit,i->bct', x, self.alpha)  # [B, C, T]
		A = torch.softmax(K @ self.W @ Q.transpose(1, 2), dim=-1)  # [B, C, C]
		# y_{c,n,t} = a_{c,i} x_{i,n,t}
		return torch.einsum('bci,bint->bcnt', A, x)  # [B, C, N, T]


class MATGCNBlock(Module):
	def __init__(self, in_channels, out_channels, in_timesteps, tcn_dilations, n_nodes, **kwargs):
		super(MATGCNBlock, self).__init__()
		self.seq = Sequential(
			LayerNorm([in_timesteps]),
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
		return torch.relu(self.seq(x) + self.res(x))  # [B, C_o, N, T]


class MATGCNLayer(Module):
	def __init__(self, blocks, **kwargs):
		super(MATGCNLayer, self).__init__()
		self.blocks = Sequential(*[MATGCNBlock(**block, **kwargs) for block in blocks])
		self.ln = LayerNorm([kwargs['in_timesteps']])
		self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'], [1, blocks[-1]['out_channels']])

	def forward(self, x: FloatTensor):
		"""
		:param x: [B, C_i, N, T_i]
		:return: [B, C_o, N, T_o]
		"""
		x = self.blocks(x)  # [B, C_o, N, T_o]
		x = self.ln(x)  # [B, C_o, N, T_o]
		x = self.fc(x.transpose(1, 3))  # [B, T_o, N, 1]
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
