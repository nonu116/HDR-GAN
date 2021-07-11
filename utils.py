import os
import sys

import tensorflow as tf

from tensorkit.annotation import wrap_tf_name_scope


@wrap_tf_name_scope()
def l1_loss(real, fake, name=None):
    assert real.get_shape().as_list() == fake.get_shape().as_list(), '{} != {}'.format(real, fake)
    with tf.name_scope(name if name is not None else ''):
        return tf.reduce_mean(tf.abs(fake - real))


@wrap_tf_name_scope()
def perceptual_loss(real, fake, weight=None):
    """
    :param real:
    :param fake:
    :param weight:
    :return:
    """
    assert real.get_shape().as_list() == fake.get_shape().as_list(), '{} != {}'.format(real, fake)
    from model.vgg import vgg_16
    end_points_keys = ['vgg_16/conv1/conv1_2',
                       'vgg_16/conv2/conv2_2',
                       'vgg_16/conv3/conv3_3',
                       'vgg_16/conv4/conv4_3',
                       'vgg_16/conv5/conv5_3']
    if weight is None:
        weight = [1.] * len(end_points_keys)
    assert len(weight) == len(end_points_keys)

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        _, r_end_points = vgg_16(real, include_top=False, is_training=False)
        _, f_end_points = vgg_16(fake, include_top=False, is_training=False)
        loss = 0.
        for w, key in zip(weight, end_points_keys):
            loss += w * l1_loss(r_end_points[key], f_end_points[key])
        return loss


LOSS_RANGE = {'L1', 'L1_HDR', 'L1_LDRS', 'L2_HDR', 'L1_LDR0',
              'LAPLACE', 'LAPLACE_HDR', 'LAPLACE_LDR0',
              'L2', 'PERCEPTUAL', 'SSIM', 'DAKR', 'TEA'}


def optimize_op(loss, var_list, optimizer, grad_clip_threshold=None, with_bn=False):
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    assert len(grads_and_vars) > 0
    if grad_clip_threshold is not None and grad_clip_threshold > 0.:
        with tf.name_scope('clip_gradient'):
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip_threshold, grad_clip_threshold), var)
                          for grad, var in grads_and_vars]
    if with_bn:
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(capped_gvs)
    else:
        train_op = optimizer.apply_gradients(capped_gvs)
    return train_op, grads_and_vars


def parse_args(flag, default=None, args=None):
    value = default
    if args is None:
        args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith(flag):
            arg = arg[len(flag):]
            if len(arg) > 0:
                if arg.startswith('='):
                    arg = arg[1:]
                value = arg
            else:
                try:
                    arg = args[i + 1]
                    if arg.startswith('-'):
                        raise
                except Exception:
                    print('{}: error: argument {}: expected one argument'.format(
                        sys.argv[0].split(os.sep)[-1],
                        flag
                    ), file=sys.stderr)
                    exit(-1)
                value = arg
    return value
