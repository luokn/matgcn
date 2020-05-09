# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import os
from datetime import datetime

import torch


def read_adj_matrix(adj_file, n_nodes, device='cpu'):
    """
    :param adj_file:
    :param n_nodes:
    :param device:
    :return: \tilde A = A + I_N
    """
    A = torch.eye(n_nodes, device=device)
    for ln in open(adj_file, 'r').readlines()[1:]:
        i, j, _ = ln.split(',')
        i, j = int(i), int(j)
        A[i, j] = A[j, i] = 1

    return A


def make_saved_dir(saved_dir):
    """
    :param saved_dir:
    :return: {saved_dir}/{%m-%d-%H-%M-%S}
    """
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d-%H-%M-%S'))
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    return saved_dir


def log_to_file(file, **kwargs):
    with open(file, 'a') as f:
        f.write(','.join([f'{k}={v}' for k, v in kwargs.items()]))
        f.write('\n')
