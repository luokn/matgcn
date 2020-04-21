import json
import sys

from tools.config import Config
from tools.trainer import Trainer

if __name__ == '__main__':
	file = sys.argv[1] if len(sys.argv) >= 2 else 'debug'
	conf = json.loads(open(f'./config/{file}.json').read())
	Trainer(Config(**conf)).run()
