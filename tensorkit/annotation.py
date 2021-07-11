from functools import wraps

import tensorflow as tf

from tensorkit.log import logger, Color


def wrap_tf_variable_scope(name):
    def _deco(func):
        @wraps(func)
        def _d(*args, **kwargs):
            with tf.variable_scope(name):
                return func(*args, **kwargs)

        return _d

    return _deco


def deprecated(msg='', prefer: callable = None):
    def _deco(func):
        @wraps(func)
        def _d(*args, **kwargs):
            if prefer is not None:
                con = Color.yellow(
                    'DEPRECATED: %s@%s is deprecated. updating: Use %s@%s' % (func.__module__, func.__name__,
                                                                              prefer.__module__, prefer.__name__),
                    bold=True)
            else:
                con = Color.yellow('DEPRECATED: %s@%s is deprecated. %s' % (func.__module__, func.__name__, msg),
                                   bold=True)
            logger.warn(con)
            return func(*args, **kwargs)

        return _d

    return _deco


def wrap_tf_name_scope():
    def _deco(func):
        @wraps(func)
        def _d(*args, **kwargs):
            with tf.name_scope(func.__name__):
                return func(*args, **kwargs)

        return _d

    return _deco
