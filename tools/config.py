# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

from dataclasses import dataclass


@dataclass
class Config:
	lr: float
	epochs: int
	batch_size: int
	data_split: float
	adj_file: str
	data_file: str
	saved_dir: str
	n_nodes: int
	out_timesteps: int
	points_per_hour: int
	device_for_data: str = 'cpu'
	device_for_model: str = 'cpu'
