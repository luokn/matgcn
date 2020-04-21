import torch


def norm_adj_matrix(adj_file, n_vertices, device='cpu'):
	A = torch.eye(n_vertices, device=device)
	for ln in open(adj_file, 'r').readlines()[1:]:
		i, j, _ = ln.split(',')
		i, j = int(i), int(j)
		A[i, j] = A[j, i] = 1

	D_rsqrt = torch.sum(A, dim=1).rsqrt().diag()
	return D_rsqrt @ A @ D_rsqrt  # D^{-1/2} A D^{-1/2}
