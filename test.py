import argparse
import importlib
import os

from config import config
from utils import parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', dest='CKPT_FILE', required=True)
    parser.add_argument('--tag', dest='TAG', default=...)
    parser.add_argument('--gpu', dest='CUDA_VISIBLE_DEVICES', default=-1, type=int)
    parser.add_argument('--ignore_config', default=False, action='store_true')
    parser.add_argument('--module', default='unetpps')
    parser.add_argument('--cus_test_ds', dest='TEST_DS', default=...)
    parser.add_argument('--test_hw', dest='test_hw', default=..., nargs=2)

    module_name = 'test_{}'.format(parse_args('--module', 'unetpps'))
    test_module = importlib.import_module(module_name)
    parser = test_module.args(parser)
    args = parser.parse_args()

    if not args.ignore_config:
        ckpt_dir = os.path.dirname(args.CKPT_FILE)
        config_file = os.path.join(ckpt_dir, 'config.yml')
        assert os.path.isfile(config_file)
        config.load(config_file)
    config.apply(args)
    test_module.test()
