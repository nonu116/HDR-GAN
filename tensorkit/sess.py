import os
from collections import Iterable

import tensorflow as tf

_default_session = None


def default_session(gpu=None, allow_growth=True):
    global _default_session
    if _default_session is None:
        _default_session = tf.get_default_session()
        if _default_session is None:
            _default_session = session(gpu, allow_growth)
    return _default_session


def session(gpu=None, allow_growth=True, per_process_gpu_memory_fraction=None):
    """
    :param gpu:
    :param allow_growth:
    :param per_process_gpu_memory_fraction:
    :return:
    """
    _gpu = '-1'
    if isinstance(gpu, int) and gpu >= 0:
        _gpu = str(gpu)
    elif isinstance(gpu, str):
        _gpu = gpu
    elif isinstance(gpu, Iterable):
        _gpu = ','.join([str(i) for i in gpu])

    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    if per_process_gpu_memory_fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = float(per_process_gpu_memory_fraction)
    return tf.Session(config=config)
