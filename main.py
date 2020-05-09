# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import sys

from tools.config import config
from tools.trainer import Trainer

if __name__ == '__main__':
    assert len(sys.argv) >= 2
    Trainer(config(sys.argv[1])).fit()
