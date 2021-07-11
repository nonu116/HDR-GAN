import os
import traceback

import tensorflow as tf

from tensorkit.config import Config
from tensorkit.log import logger, Color

__hyper_parameters = dict()


class HParameter(object):

    def __init__(self, name, shape=None, initializer=None, dtype=None) -> None:
        super().__init__()
        with tf.variable_scope(name):
            self.name = name
            self.var = tf.get_variable(name='val', shape=shape,
                                       initializer=initializer, dtype=dtype, trainable=False)
            self.ph = tf.placeholder(name='ph', dtype=self.var.dtype, shape=self.var.get_shape())
            self.assign = tf.assign(self.var, self.ph)
        self.sync_key = None
        self.sync_file = None

    def set_sync(self, key, file_name):
        self.sync_key = key
        self.sync_file = file_name


def get_hparamter(name, shape=None, initializer=None, dtype=None):
    param = HParameter(name, shape, initializer, dtype)
    __hyper_parameters[param.var.name] = param
    return param.var


def set_sync(hparam, key, file_name):
    ph = __hyper_parameters[hparam.name]
    ph.set_sync(key, file_name)


def sync_hparamters(sess: tf.Session):
    sync_files = []
    try:
        assign_ops = []
        phs = []
        sync_keys = []
        hps = [hp for hp in __hyper_parameters.values() if os.path.isfile(hp.sync_file)]
        for hp in hps:
            if os.path.isfile(hp.sync_file):
                assign_ops.append(hp.assign)
                phs.append(hp.ph)
                sync_keys.append(hp.sync_key)
                sync_files.append(hp.sync_file)

        source_data = dict()
        for f in set(sync_files):
            cfg = Config(f)
            source_data[f] = cfg
        new_vals = [source_data[f].safely_get(k, None) for k, f in zip(sync_keys, sync_files)]

        ops = []
        fd = dict()
        _hps = []
        _vals = []
        for ao, ph, val, hp in zip(assign_ops, phs, new_vals, hps):
            if val is not None:
                ops.append(ao)
                fd[ph] = val
                _hps.append(hp)
                _vals.append(val)

        if len(ops) == 0:
            return
        sess.run(ops, feed_dict=fd)
        content = [' update '.center(50, '=')]
        content.extend([Color.yellow('{} -> {}'.format(hp.name, val), bold=True) for hp, val in zip(_hps, _vals)])
        content.append(''.center(50, '='))
        print()
        logger.info('\n'.join(content))
        for fn in set(sync_files):
            os.rename(fn, '{}.succeed'.format(fn))
    except:
        content = [' update '.center(50, '='),
                   traceback.format_exc(),
                   'config files: {}',
                   ''.center(50, '=')]
        print()
        logger.info('\n'.join(content))
        for fn in set(sync_files):
            os.rename(fn, '{}.error'.format(fn))
