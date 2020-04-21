class Config:
	lr: float
	epochs: int
	batch_size: int
	data_split: float
	adj_file: str
	data_file: str
	saved_dir: str
	n_vertices: int
	in_timesteps: int
	out_timesteps: int
	points_per_hour: int
	device_for_data: str = 'cpu'
	device_for_model: str = 'cpu'

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
