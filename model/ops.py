import tensorflow as tf

he_normal = tf.keras.initializers.he_normal
conv2d_transpose = tf.layers.conv2d_transpose


def swish(input_tensor):
    return input_tensor * tf.nn.sigmoid(input_tensor)


def spectral_norm(w, iteration=1):
    """
    https://github.com/taki0112/Spectral_Normalization-Tensorflow
    :param w:
    :param iteration:
    :return:
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_, [0, 1])

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_, [0, 1])

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def weight_norm(w):
    w_shape = w.get_shape().as_list()
    assert len(w_shape) == 4  # [ks, ks, ci, co]
    v = w
    g_init = tf.norm(tf.reshape(v, [-1, w_shape[-1]]), axis=0)
    g_init = tf.reshape(g_init, [1, 1, 1, w_shape[-1]])
    g = tf.get_variable('wn_g', initializer=g_init)
    v_norm = tf.nn.l2_normalize(w, axis=[0, 1, 2])
    return g * v_norm


def conv2d(inp, filters, kernel_size, strides=(1, 1), padding='reflect', dilation_rate=1, activation=None,
           trainable=True, name=None, name_key=None, b_init=0.0, kernel_init_hook=None, bias_init_hook=None):
    assert name is not None or name_key is not None
    assert padding in ['reflect', 'valid', 'same'], padding
    assert dilation_rate in [1, 2], dilation_rate
    x = inp
    padding = padding.lower()
    _ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    _sts = (strides, strides) if isinstance(strides, int) else strides
    _name = name if name_key is None else 'cov_{}'.format(name_key)
    with tf.variable_scope(_name):
        if padding == 'reflect':
            p = [int((k - 1) / 2) + dilation_rate - 1 for k in _ks]
            if any(_pi > 0 for _pi in p):
                x = tf.pad(x, [[0, 0], p, p, [0, 0]], mode='REFLECT', name='reflect_padding')
            padding = 'valid'

        w_shape = tf.TensorShape([_ks[0], _ks[1], inp.get_shape().as_list()[3], filters])
        b_shape = [filters]
        if kernel_init_hook is not None:
            w = kernel_init_hook(w_shape)
        else:
            w = tf.get_variable(shape=w_shape, initializer=he_normal(), trainable=trainable, name='kernel')
        if bias_init_hook is not None:
            b = bias_init_hook(b_shape)
        else:
            b = tf.get_variable(name='bias', shape=[filters], trainable=trainable,
                                initializer=tf.constant_initializer(b_init))
        out = tf.nn.conv2d(x, filter=w,
                           strides=[1, _sts[0], _sts[1], 1],
                           dilations=[1, dilation_rate, dilation_rate, 1],
                           padding=padding.upper()) + b
        if activation is not None:
            out = activation(out)
    return out


def sn_kernel_init_hook(shape):
    w = tf.get_variable(shape=shape, trainable=True, name='kernel', initializer=he_normal())
    return spectral_norm(w)


def wn_kernel_init_hook(shape):
    w = tf.get_variable(shape=shape, trainable=True, name='kernel',
                        initializer=tf.truncated_normal_initializer(0., 0.05))
    return weight_norm(w)


def sn_conv2d(inp, filters, kernel_size, strides=(1, 1), padding='reflect', dilation_rate=1, activation=None,
              trainable=True, name=None, name_key=None, b_init=0.0):
    return conv2d(inp=inp, filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, dilation_rate=dilation_rate, activation=activation,
                  trainable=trainable, name=name, name_key=name_key,
                  b_init=b_init, kernel_init_hook=sn_kernel_init_hook, bias_init_hook=None)


def wn_conv2d(inp, filters, kernel_size, strides=(1, 1), padding='reflect', dilation_rate=1, activation=None,
              trainable=True, name=None, name_key=None, b_init=0.0):
    return conv2d(inp=inp, filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, dilation_rate=dilation_rate, activation=activation,
                  trainable=trainable, name=name, name_key=name_key,
                  b_init=b_init, kernel_init_hook=wn_kernel_init_hook, bias_init_hook=None)


def down2_sample_mp(inp, name=None, name_key=None):
    name = 'pool_{}'.format(name_key) if name_key is not None else name
    return tf.layers.max_pooling2d(inp, (2, 2), (2, 2), padding='valid', name=name)


def down2_sample_conv(inp, kernel_size, activation=None, name=None, name_key=None, kernel_init_hook=None,
                      bias_init_hook=None):
    c = inp.get_shape().as_list()[-1]
    name = 'down_{}'.format(name_key) if name_key is not None else name
    return conv2d(inp, c, kernel_size, 2, activation=activation, name=name,
                  kernel_init_hook=kernel_init_hook, bias_init_hook=bias_init_hook)


def up2_sample_transpose(inp, kernel_size, filters=None, activation=None, name=None, name_key=None):
    name = 'up_{}'.format(name_key) if name_key is not None else name
    _, h, w, c = inp.get_shape().as_list()
    filters = c if filters is None else filters
    return tf.layers.conv2d_transpose(inp, filters=filters, kernel_size=kernel_size, strides=2,
                                      kernel_initializer=he_normal(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                      activation=activation, padding='same', name=name)


def up_sample_resize(inp, kernel_size, oh, ow, filters=None, activation=None, name=None, name_key=None,
                     kernel_init_hook=None, bias_init_hook=None):
    name = 'up_{}'.format(name_key) if name_key is not None else name
    _, h, w, c = inp.get_shape().as_list()
    assert any(i is None for i in (oh, h, ow, w)) or oh != h or ow != w, '{} == {} or {} == {}'.format(oh, h, ow, w)
    filters = c if filters is None else filters
    x = tf.image.resize_nearest_neighbor(inp, (oh, ow), align_corners=True, name=name + '_resize')
    return conv2d(x, filters, kernel_size, 1, padding='reflect', activation=activation, name=name,
                  kernel_init_hook=kernel_init_hook, bias_init_hook=bias_init_hook)


def up2_sample_resize(inp, kernel_size, filters=None, activation=None, name=None, name_key=None,
                      kernel_init_hook=None, bias_init_hook=None):
    _, h, w, c = inp.get_shape().as_list()
    if h is None or w is None:
        shape = tf.shape(inp)
        h, w = shape[1], shape[2]
    oh, ow = h * 2, w * 2
    return up_sample_resize(inp, kernel_size, oh, ow, filters, activation, name, name_key,
                            kernel_init_hook, bias_init_hook)


def batch_norm(inp, training, name=None, name_key=None):
    name = 'bn_{}'.format(name_key) if name_key is not None else name
    return tf.layers.batch_normalization(inp, training=training, fused=True, name=name)


def instance_norm(inp, training, name=None, name_key=None):
    """
        tf.contrib.layers.instance_norm(
            inputs,
            center=True,
            scale=True,
            epsilon=1e-06,
            activation_fn=None,
            param_initializers=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            data_format=DATA_FORMAT_NHWC,
            scope=None
        )
    :param inp:
    :param training:
    :param name:
    :return:
    """
    name = '' if name is None else name
    name = 'in_{}'.format(name_key) if name_key is not None else name
    with tf.variable_scope(name):
        return tf.contrib.layers.instance_norm(inp)


def layer_norm(inp, training, name=None, name_key=None):
    """
    :param inp:
    :param training:
    :param name:
    :return:
    """
    name = '' if name is None else name
    name = 'ln_{}'.format(name_key) if name_key is not None else name
    with tf.variable_scope(name):
        return tf.contrib.layers.layer_norm(inp)


def none_norm(inp, training, name=None, name_key=None):
    return inp


def standard_res_unit(input_tensor, filters, kernel_size=3, strides=1, act=tf.nn.relu, training=True,
                      norm=instance_norm, name=None, name_key=None, kernel_init_hook=None, bias_init_hook=None):
    """
    :param input_tensor:
    :param filters:
    :param kernel_size:
    :param strides:
    :param act:
    :param training:
    :param norm:
    :param name:
    :param name_key:
    :param kernel_init_hook:
    :param bias_init_hook:
    :return:
    """
    x = input_tensor
    name = '' if name is None else name
    name = name if name_key is None else 'SR_{}'.format(name_key)
    with tf.variable_scope(name):
        x = conv2d(x, filters, kernel_size, strides=1, name_key='0',
                   kernel_init_hook=kernel_init_hook, bias_init_hook=bias_init_hook)
        x = norm(x, training=training, name_key='0')
        x = act(x) if act is not None else x

        x = conv2d(x, filters, kernel_size, strides=strides, name_key='1',
                   kernel_init_hook=kernel_init_hook, bias_init_hook=bias_init_hook)

        in_c = input_tensor.get_shape().as_list()[-1]
        if in_c != filters:
            input_tensor = conv2d(input_tensor, filters, 1, 1, name_key='btn',
                                  kernel_init_hook=kernel_init_hook, bias_init_hook=bias_init_hook)
        x = tf.add(x, input_tensor, name='res')
        x = norm(x, training=training, name_key='1')
        x = act(x) if act is not None else x
    return x


def _main():
    pass


if __name__ == '__main__':
    _main()
