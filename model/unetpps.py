import tensorflow as tf

import tensorkit as tk
from model import Unit
from model.ops import conv2d as _conv2d
from model.ops import down2_sample_mp, down2_sample_conv
from model.ops import instance_norm, layer_norm, none_norm
from model.ops import sn_conv2d, wn_conv2d
from model.ops import sn_kernel_init_hook, wn_kernel_init_hook
from model.ops import standard_res_unit as _standard_res_unit
from model.ops import up2_sample_resize
from tensorkit import BaseModel


class UnetppGeneratorS(BaseModel):
    """
    modify act and channels and bottle_neck
    """

    def __init__(self, depth=5, down_conv=True, name=None, one_encoder=False,
                 stop_gradient=False, act='relu', no_decoders=False, use_bottle_neck=False, norm='sn'):
        assert depth in [2, 3, 4, 5], 'depth: {}'.format(depth)
        assert act in ['relu', 'swish'], 'invalid act: {}'.format(act)
        assert (stop_gradient and one_encoder) or not stop_gradient

        # self.merge = merge
        self.depth = depth
        self.down_conv = down_conv
        self.one_encoder = one_encoder
        self.stop_gradient = stop_gradient
        self.act = tf.nn.swish if act == 'swish' else tf.nn.relu
        self.no_decoders = no_decoders
        self.use_bottle_neck = use_bottle_neck
        self._init_normalization(norm)
        super().__init__('UnetppGeneratorS')

    def _init_normalization(self, norm):
        assert norm in ['in', 'ln', 'nn', 'sn', 'wn'], 'invalid norm: {}'.format(norm)
        self.conv2d = _conv2d
        self.kernel_init_hook = None
        self.norm = none_norm
        if norm == 'in':
            self.norm = instance_norm
        elif norm == 'ln':
            self.norm = layer_norm
        elif norm == 'sn':
            self.conv2d = sn_conv2d
            self.kernel_init_hook = sn_kernel_init_hook
        elif norm == 'wn':
            self.conv2d = wn_conv2d
            self.kernel_init_hook = wn_kernel_init_hook

    def standard_res_unit(self, input_tensor, filters, kernel_size=3, strides=1, act=None, training=True,
                          name=None, name_key=None):
        if act is None:
            act = self.act
        return _standard_res_unit(input_tensor, filters, kernel_size=kernel_size, strides=strides,
                                  act=act, training=training, norm=self.norm, name=name, name_key=name_key,
                                  kernel_init_hook=self.kernel_init_hook)

    def encoder(self, x, filters, training=True):
        assert len(filters) >= self.depth
        act = self.act
        standard_res_unit = self.standard_res_unit

        features, stage_units = [], []
        for ind in range(self.depth):
            unit = Unit(inp=x)
            x = standard_res_unit(x, filters[ind], act=act, training=training, name_key=ind)
            # unit.x = standard_res_unit(x, filters[ind], act=act, name='su_%d1' % ind)
            unit.x = x
            if ind != self.depth - 1:
                if self.down_conv:
                    unit.d = down2_sample_conv(unit.x, 3, name_key=ind,
                                               kernel_init_hook=self.kernel_init_hook)
                    unit.d = self.act(self.norm(unit.d, training=training, name_key=ind))
                else:
                    unit.d = down2_sample_mp(unit.x, name_key=ind)
            x = unit.d
            features.append(unit.x)
            stage_units.append(unit)
        return features, stage_units

    def bottle_neck(self, inp, filters, name_key, training=True):
        inp_c = inp.get_shape().as_list()[-1]
        x = inp
        if self.use_bottle_neck and inp_c != filters:
            x = self.conv2d(x, filters, 1, 1, 'valid', trainable=training, name_key=name_key)
            x = self.norm(x, training=training, name_key=name_key)
            x = self.act(x)
        return x

    def graph(self, image1, image2, image3, train=True, reuse=None, summary_feat=False, get_features=False,
              image1_ref=None, image3_ref=None):
        """
        :return: outputs(list), features(dict)
        """
        act = self.act
        standard_res_unit = self.standard_res_unit
        with tf.variable_scope(self.name, reuse=reuse):
            filters = [32, 64, 128, 256, 256]

            stages = [[] for _ in range(5)]

            with tf.variable_scope('backbone_1' if not self.one_encoder else 'backbone_0'):
                features1, stage_ref = self.encoder(image2, filters)

            _reuse = True if self.one_encoder else None
            with tf.variable_scope('backbone_0', reuse=_reuse):
                features0, stage_00 = self.encoder(image1, filters)

            with tf.variable_scope('backbone_2' if not self.one_encoder else 'backbone_0', reuse=_reuse):
                features2, stage_02 = self.encoder(image3, filters)

            features0_ref, stage_00_ref, features2_ref, stage_02_ref = [None] * 4
            if image1_ref is not None:
                with tf.variable_scope('backbone_0', reuse=True):
                    features0_ref, stage_00_ref = self.encoder(image1_ref, filters)
            if image3_ref is not None:
                with tf.variable_scope('backbone_2' if not self.one_encoder else 'backbone_0', reuse=True):
                    features2_ref, stage_02_ref = self.encoder(image3_ref, filters)

            feature_merge = []
            deform_features0, deform_features2 = [], []
            transf_features0, transf_features2 = [], []
            image1_def, image2_def, w1, w2 = [None] * 4
            with tf.variable_scope('merge'):
                for ind, us in enumerate(zip(stage_00, stage_ref, stage_02)):
                    u0, u_ref, u2 = us
                    if self.stop_gradient:
                        u0.x = tf.stop_gradient(u0.x)
                        u2.x = tf.stop_gradient(u2.x)
                    if summary_feat:
                        tk.summary.histogram(u0.x, 'backbone_0_%d' % ind)
                        tk.summary.histogram(u_ref.x, 'backbone_1_%d' % ind)
                        tk.summary.histogram(u2.x, 'backbone_2_%d' % ind)
                    unit = Unit()
                    unit.x = u_ref.x  # for skip connection
                    unit.mx = merge_net(u0.x, u_ref.x, u2.x, training=train, act=self.act,
                                        norm=self.norm, conv2d=self.conv2d, name_key=ind)
                    feature_merge.append(unit.mx)
                    if ind != 0:
                        unit.u = up2_sample_resize(unit.mx, 3, filters=filters[ind - 1], activation=act,
                                                   name_key=ind)
                    stages[0].append(unit)

            if summary_feat:
                def unit_feat_shape(features):
                    assert type(features) in [list, tuple], 'invalid input type:{}'.format(type(features))
                    shape0 = features[0].get_shape().as_list()[1:3]
                    return [tf.image.resize_nearest_neighbor(i, shape0, True)
                            if i.get_shape().as_list()[1:3] != shape0 else i for i in features]

                ffs = [features0, features1, features2]
                if len(deform_features0) != 0:
                    assert (len(features0) - 1) == len(deform_features0) == len(deform_features2), \
                        '{}!={}'.format(len(features0), len(deform_features0))
                    c = len(ffs)
                    ffs = ffs[:c // 2] + [[features0[0]] + deform_features0,
                                          ffs[c // 2], [features2[0]] + deform_features2] + ffs[c // 2 + 1:]
                if features0_ref is not None:
                    c = len(ffs)
                    ffs = ffs[:c // 2] + [features0_ref, ffs[c // 2], features2_ref] + ffs[c // 2 + 1:]

                if len(transf_features0) != 0:
                    assert (len(features0) - 1) == len(transf_features0) == len(transf_features2), \
                        '{}!={}'.format(len(features0), len(transf_features0))
                    c = len(ffs)
                    ffs = ffs[:c // 2] + [[features0[0]] + transf_features0,
                                          ffs[c // 2], [features2[0]] + transf_features2] + ffs[c // 2 + 1:]

                means = [[tf.reduce_mean(i, axis=-1, keepdims=True) for i in features]
                         for features in ffs]
                maxs = [[tf.reduce_max(i, axis=-1, keepdims=True) for i in features]
                        for features in ffs]

                def fsg2image(fsg):
                    if len(fsg) > 3:
                        c = len(fsg)
                        _ffs1, _ffs2 = fsg[:c // 2], fsg[c // 2 + 1:]
                        fs1 = [tf.concat(i, axis=1) for i in zip(*_ffs1)]
                        fs2 = [tf.concat(i, axis=1) for i in zip(*_ffs2)]
                        fsg = [fs1, fsg[c // 2], fs2]
                    fsg_color = [[tk.image.colorize(f) for f in features] for features in fsg]
                    fsg_color = [unit_feat_shape(i) for i in fsg_color]
                    fsg_color = [tf.concat(i, axis=2) for i in fsg_color]
                    fsg_color = tf.concat(fsg_color, axis=1)
                    return fsg_color

                means_color = fsg2image(means)
                maxs_color = fsg2image(maxs)
                tk.summary.images(means_color, 3, name='mean/f0_f0d_f0r_f0t_f1_f2t_f2r_f2d_f2')
                tk.summary.images(maxs_color, 3, name='max/f0_f0d_f0r_f0t_f1_f2t_f2r_f2d_f2')

            if self.no_decoders:
                return None, {
                    'features0': features0,
                    'features1': features1,
                    'features2': features2,
                    'features0_ref': features0_ref,
                    'features2_ref': features2_ref,
                    'image1_def': image1_def,
                    'image2_def': image2_def,
                    'deform_features0': deform_features0,
                    'deform_features2': deform_features2,
                    'transf_features0': transf_features0,
                    'transf_features2': transf_features2,
                    'feature_merge': feature_merge
                }
            with tf.variable_scope('decoders'):
                stage_number = self.depth - 1  # ds_0
                for ind in range(stage_number):
                    unit = Unit()
                    inp = [stages[0][ind + 1].u]  # upsample
                    inp += [stages[0][ind].mx]
                    inp += [stages[0][ind].x]  # skip
                    unit.inp = tf.concat(inp, axis=-1)
                    unit.inp = self.bottle_neck(unit.inp, filters[ind], 'btn_1{}'.format(ind), train)
                    unit.x = standard_res_unit(unit.inp, filters[ind], 3, 1, act=act, training=train,
                                               name_key='1{}'.format(ind))
                    if ind != 0:
                        unit.u = up2_sample_resize(unit.x, 3, filters=filters[ind - 1], activation=act,
                                                   name_key='1{}'.format(ind))
                    stages[1].append(unit)

                stage_number = self.depth - 2  # ds_1
                for ind in range(stage_number):
                    unit = Unit()
                    inp = [stages[1][ind + 1].u]
                    inp += [stages[0][ind].mx]
                    inp += [stages[0][ind].x, stages[1][ind].x]  # skip connection
                    unit.inp = tf.concat(inp, axis=-1)
                    unit.inp = self.bottle_neck(unit.inp, filters[ind], 'btn_2{}'.format(ind), train)
                    unit.x = standard_res_unit(unit.inp, filters[ind], 3, 1, act=act, training=train,
                                               name_key='2{}'.format(ind))
                    if ind != 0:
                        unit.u = up2_sample_resize(unit.x, 3, filters=filters[ind - 1], activation=act,
                                                   name_key='2{}'.format(ind))
                    stages[2].append(unit)

                stage_number = self.depth - 3  # ds_2
                for ind in range(stage_number):
                    unit = Unit()
                    inp = [stages[2][ind + 1].u]
                    inp += [stages[0][ind].mx]
                    inp += [stages[0][ind].x, stages[1][ind].x, stages[2][ind].x]
                    unit.inp = tf.concat(inp, axis=-1)
                    unit.inp = self.bottle_neck(unit.inp, filters[ind], 'btn_3{}'.format(ind), train)
                    unit.x = standard_res_unit(unit.inp, filters[ind], 3, 1, act=act, training=train,
                                               name_key='3{}'.format(ind))
                    if ind != 0:
                        unit.u = up2_sample_resize(unit.x, 3, filters=filters[ind - 1], activation=act,
                                                   name_key='3{}'.format(ind))
                    stages[3].append(unit)

                stage_number = self.depth - 4  # ds_3
                for ind in range(stage_number):
                    unit = Unit()
                    inp = [stages[3][ind + 1].u]
                    inp += [stages[0][ind].mx]
                    inp += [stages[ind][0].x for ind in range(4)]
                    unit.inp = tf.concat(inp, axis=-1)
                    unit.inp = self.bottle_neck(unit.inp, filters[ind], 'btn_4{}'.format(ind), train)
                    unit.x = standard_res_unit(unit.inp, filters[ind], 3, 1, act=act, training=train,
                                               name_key='4{}'.format(ind))
                    if ind != 0:
                        unit.u = up2_sample_resize(unit.x, 3, filters=filters[ind - 1], activation=act,
                                                   name_key='4{}'.format(ind))
                    stages[4].append(unit)

            outputs = []
            for i in range(self.depth - 1):
                output = self.conv2d(stages[i + 1][0].x, 3, 1, 1, activation=tf.nn.tanh, name='output_{}'.format(i + 1))
                outputs.append(output)
            ret_features = {
                'features0': features0,
                'features1': features1,
                'features2': features2,
                'features0_ref': features0_ref,
                'features2_ref': features2_ref,
                'image1_def': image1_def,
                'image2_def': image2_def,
                'deform_features0': deform_features0,
                'deform_features2': deform_features2,
                'transf_features0': transf_features0,
                'transf_features2': transf_features2,
                'feature_merge': feature_merge
            }
            if not get_features:
                ret_features = None
            return outputs, ret_features

    def variable_merge(self):
        vars_all = self.all_variables()
        res = []
        for v in vars_all:
            if 'merge' in v.name:
                res.append(v)
        return res

    def variable_backbone0(self):
        res = []
        vars_all = self.all_variables()
        for v in vars_all:
            if 'backbone_0' in v.name:
                res.append(v)
        return res

    def variable_backbone1(self):
        res = []
        vars_all = self.all_variables()
        for v in vars_all:
            if 'backbone_1' in v.name:
                res.append(v)
        return res

    def variable_backbone2(self):
        res = []
        vars_all = self.all_variables()
        for v in vars_all:
            if 'backbone_2' in v.name:
                res.append(v)
        return res

    def variable_decoders(self):
        vars_all = self.all_variables()
        res = []
        tags = ['merge', 'backbone_0', 'backbone_1', 'backbone_2']
        for v in vars_all:
            if all([tag not in v.name for tag in tags]):
                res.append(v)
        return res

    def variable_out(self):
        res = []
        vars_all = self.all_variables()
        for v in vars_all:
            if 'output_' in v.name:
                res.append(v)
        return res


# conv2d = _conv2d


def merge_net(inp1, inp_ref, inp2, training, act, norm, conv2d, name_key):
    _inp_ref = inp_ref
    conv_c = _inp_ref.get_shape().as_list()[-1]
    with tf.variable_scope('merge_net_{}'.format(name_key)):
        if inp_ref.get_shape().as_list()[-1] * 3 > 512:
            inp1 = conv2d(inp1, 170, 1, 1, padding='valid', activation=act, name='blt_1')
            inp_ref = conv2d(inp_ref, 170, 1, 1, padding='valid', activation=act, name='blt_r')
            inp2 = conv2d(inp2, 170, 1, 1, padding='valid', activation=act, name='blt_2')
        x = tf.concat([inp1, inp_ref, inp2], axis=-1)
        x = conv2d(x, conv_c, 3, 1, dilation_rate=2, name='cov_1')
        x = act(norm(x, training=training, name='norm_1'))
        x = conv2d(x, conv_c, 3, 1, dilation_rate=2, name='cov_2')
        x = act(norm(x, training=training, name='norm_2'))
        x = conv2d(x, conv_c, 3, 1, dilation_rate=2, name='cov_3')
        x = act(norm(x, training=training, name='norm_3'))
        x = conv2d(x, conv_c, 3, 1, dilation_rate=2, name='cov4')
        x = tf.add(x, _inp_ref, name='res')
        x = act(norm(x, training=training, name='in4'))
    return x
