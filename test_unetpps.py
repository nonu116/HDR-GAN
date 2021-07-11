import os
import sys

import cv2
import numpy as np
import tensorflow as tf

import hdr_utils
import tensorkit as tk
from config import config
from model.unetpps import UnetppGeneratorS
from tensorkit import logger, logging_to_file

UnetGeneratorS, UnetppGenerator = None, None


class TestData(object):

    def __init__(self) -> None:
        scenes = sorted(list(os.listdir(config.TEST_DS)))
        # assert len(scenes) == config.TEST_SIZE
        self.scenes = [os.path.join(config.TEST_DS, i) for i in scenes]
        self.image_size = config.test_hw

    def _center_crop(self, x):
        crop_h, crop_w = self.image_size
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        x = x[max(0, j):min(h, j + crop_h), max(0, i):min(w, i + crop_w), :]
        if x.shape[:2] != (crop_h, crop_w):
            x = cv2.resize(x, (crop_w, crop_h))
        return x

    def __getitem__(self, item):
        scene = self.scenes[item]
        ldr = ['input_{}.tif'.format(i + 1) for i in range(3)]
        ldr = [os.path.join(scene, i) for i in ldr]
        ldr = [tk.image.read_image_np(i) for i in ldr]
        hdr_ref = ldr[0]
        has_reference = os.path.isfile(os.path.join(scene, 'ref_hdr.hdr'))
        if has_reference:
            hdr_ref = hdr_utils.read_hdr(os.path.join(scene, 'ref_hdr.hdr'))
        images = np.concatenate(ldr + [hdr_ref], axis=-1)
        images = self._center_crop(images)
        images = np.expand_dims(images, 0)
        ldr1, ldr2, ldr3, hdr_ref = np.split(images, 4, axis=-1)
        if not has_reference:
            hdr_ref = None
        with open(os.path.join(scene, 'input_exp.txt')) as f:
            exps = f.read().split('\n')
            exps = np.array([float(i) for i in exps[:3]], dtype=np.float32)
            exps -= exps.min()
            exps = 2.0 ** np.reshape(exps, (-1, 3))
        return ldr1, ldr2, ldr3, exps, hdr_ref

    def __len__(self):
        return len(self.scenes)


def graph():
    ldr1, ldr2, ldr3 = [tf.placeholder(tf.float32, [None, None, None, 3]) for _ in range(3)]
    exps = tf.placeholder(tf.float32, [None, 3])
    hdr1, hdr2, hdr3 = [hdr_utils.ldr2hdr(ldr, tf.reshape(exps[..., ei], [-1, 1, 1, 1]))
                        for ldr, ei in zip([ldr1, ldr2, ldr3], range(3))]
    tp1, tp2, tp3 = [hdr_utils.tonemap(hdr) for hdr in [hdr1, hdr2, hdr3]]

    im1 = tf.concat([ldr1, hdr1 if config.IN_HDR else tp1], axis=-1)
    im2 = tf.concat([ldr2, hdr2 if config.IN_HDR else tp2], axis=-1)
    im3 = tf.concat([ldr3, hdr3 if config.IN_HDR else tp3], axis=-1)
    if config.UNETPPS:
        assert config.GENERATOR in ['', 'unetpps']
        config.GENERATOR = 'unetpps'
    if config.GENERATOR == 'unetpps':
        generator = UnetppGeneratorS
    elif config.GENERATOR == 'unets':
        generator = UnetGeneratorS
    elif config.GENERATOR in ['', 'unetpp']:
        generator = UnetppGenerator
    else:
        raise NotImplementedError('generator: {}'.format(config.GENERATOR))
    model = generator(depth=config.DEPTH, norm=config.NORM)
    outputs, _ = model.graph(im1, im2, im3, train=False, summary_feat=False, get_features=False)
    if config.OUT_HDR:
        outputs_tp = [hdr_utils.tonemap(i, mu=config.MU) for i in outputs]
    else:
        outputs_tp = outputs
    return outputs_tp, ldr1, ldr2, ldr3, exps


def save_result(outputs, real_hdr, file_name):
    for i in range(len(outputs)):
        fake_tp = outputs[i]
        fake_hdr = hdr_utils.itonemap_np(fake_tp)
        tk.image.save_image(fake_tp, '{}_fake_tp_{}.png'.format(file_name, i))
        hdr_utils.write_hdr('{}_fake_hdr_{}.hdr'.format(file_name, i), (fake_hdr + 1.) / 2.)
    if real_hdr is not None:
        hdr_utils.write_hdr('{}_real_hdr.hdr'.format(file_name), (real_hdr + 1.) / 2.)


def test():
    test_data = TestData()
    log_dir = '{}_{}_{}'.format(tk.utils.get_time(), os.getpid(), config.safely_get('TAG', ''))
    log_dir = os.path.join(config.TEST_DIR, log_dir.strip('_'))
    logging_to_file(os.path.join(log_dir, 'log'), False)
    logger.info('CMD: {}'.format(' '.join(sys.argv)))

    outputs_tp, ldr1_ph, ldr2_ph, ldr3_ph, exps_ph = graph()
    tic = tk.TimeTic()

    with tk.session(config.CUDA_VISIBLE_DEVICES) as sess:
        tk.Restore().init(ckpt_file=config.CKPT_FILE, optimistic=True).restore(sess)
        tf.get_default_graph().finalize()

        for ind, (ldr1, ldr2, ldr3, exps, real_hdr) in enumerate(test_data):
            tic.tic(1)
            outputs = sess.run(outputs_tp, {ldr1_ph: ldr1, ldr2_ph: ldr2, ldr3_ph: ldr3,
                                            exps_ph: exps})

            print('\r test {}/{}, tf_tic: {:.4f}'.format(ind, len(test_data), tic.tic(1)), end='')

            save_result(outputs, real_hdr, os.path.join(log_dir, '{:0>3d}'.format(ind)))


def args(parser):
    parser.add_argument('--in_hdr', dest='IN_HDR', default=False, action='store_true')
    parser.add_argument('--out_hdr', dest='OUT_HDR', default=False, action='store_true')
    parser.add_argument('--unetpps', dest='UNETPPS', default=False, action='store_true')
    parser.add_argument('--gen', dest='GENERATOR', default='', choices=('unetpps', 'unetpp', 'unets', 'unet'))
    parser.add_argument('--mu', dest='MU', default=None, type=float)
    return parser
