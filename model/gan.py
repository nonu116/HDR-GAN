import tensorflow as tf

import tensorkit as tk
from model.ops import conv2d as _conv2d
from model.ops import none_norm, batch_norm, layer_norm
from model.ops import sn_conv2d
from model.ops import swish


class PatchDiscriminator(tk.BaseModel):
    """
    70x70 PatchGAN with spectral_norm
    """

    def __init__(self, cnum=(64, 128, 128, 128), ori=False, name=None):
        super().__init__(name)
        assert cnum is None or len(cnum) == 4
        self.cnum = cnum
        self.act = swish
        self.conv2d = sn_conv2d
        self.norm = none_norm
        if ori:
            self.cnum = (64, 128, 256, 512)
            self.act = tf.nn.leaky_relu
            self.conv2d = _conv2d
            self.norm = batch_norm
            # TODO: diff
            self.norm = layer_norm

    def graph(self, x, reuse=None, training=True):
        with tf.variable_scope(self.name, reuse=reuse):
            cnum = self.cnum
            act = self.act
            conv2d = self.conv2d
            norm = self.norm

            x = conv2d(x, cnum[0], 4, 2, padding='valid', name='dis_cov1')
            x = act(x)

            x = conv2d(x, cnum[1], 4, 2, padding='valid', name='dis_cov2')
            x = norm(x, training, name_key='2')
            x = act(x)

            x = conv2d(x, cnum[2], 4, 2, padding='valid', name='dis_cov3')
            x = norm(x, training, name_key='3')
            x = act(x)

            x = conv2d(x, cnum[3], 4, 1, padding='valid', name='dis_cov4')
            x = norm(x, training, name_key='4')
            x = act(x)

            x = conv2d(x, 1, 4, 1, name='dis_cov5')
        return x

    def variable_out(self):
        res = []
        vars_all = self.all_variables()
        for v in vars_all:
            if 'dis_cov5' in v.name:
                res.append(v)
        return res
