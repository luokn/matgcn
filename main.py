# -*- coding: utf-8 -*-
# @Date  : 2020/4/27
# @Author: Luokun
# @Email : olooook@outlook.com

import json
import sys

from tools.config import Config
from tools.trainer import Trainer

if __name__ == '__main__':
	assert len(sys.argv) >= 2
	conf = json.loads(open(f'./config/{sys.argv[1]}.json').read())
	Trainer(Config(**conf)).fit()
