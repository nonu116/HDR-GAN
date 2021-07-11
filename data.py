import os

import numpy as np
import tensorflow as tf

import hdr_utils
import tensorkit as tk
from config import config
from tensorkit.annotation import wrap_tf_name_scope


def _get_exps(exps_file):
    with open(exps_file) as fr:
        exps = fr.read().split('\n')[:3]
    return [2. ** float(i) for i in exps]


def _get_scene_data(im_fns, motion_data: bool, aug: bool, shuffle: bool, repeat: bool):
    """
    :param im_fns:
    :param motion_data:
    :param aug:
    :param shuffle:
    :param repeat:
    :return: ims, inp_exps, ref_exps
    """
    scenes = [os.path.join(config.TRAIN_DS, i) for i in os.listdir(config.TRAIN_DS)]
    scenes = [i for i in scenes if os.path.isdir(i)]
    assert len(scenes) == config.TRAIN_SIZE
    images_count = len(im_fns)

    def scene2images(scene):
        def _py_func(_scene):
            _scene = _scene.decode('utf-8')
            images = []
            for fn in im_fns:
                if any(fn.lower().endswith(i) for i in ['.tif', '.png', '.jpg']):
                    images.append(tk.image.read_image_np(os.path.join(_scene, fn)))
                elif fn.lower().endswith('.hdr'):
                    images.append(hdr_utils.read_hdr(os.path.join(_scene, fn)))
                else:
                    raise RuntimeError('invalid fn: {}'.format(fn))
            images = np.concatenate(images, axis=-1)
            if motion_data:
                with open(os.path.join(_scene, 'motion_pos.txt')) as f:
                    x, y, h, w = [int(i) for i in f.read().split('\n')[:4]]
                images = images[x:x + h, y:y + w, :]
                images = tk.image.zoom_image_np(images, config.train_hw[0], config.train_hw[1])

            inp_exps = _get_exps(os.path.join(_scene, 'input_exp.txt'))
            ref_exps = _get_exps(os.path.join(_scene, 'ref_exp.txt'))
            assert images.shape[-1] == 3 * images_count
            return (images.astype(np.float32), np.array(images.shape, np.int32),
                    np.array(inp_exps, dtype=np.float32), np.array(ref_exps, dtype=np.float32))

        res = tf.py_func(_py_func, inp=[scene], Tout=[tf.float32, tf.int32, tf.float32, tf.float32])
        images_tf, images_shape, inp_exps_tf, ref_exps_tf = res
        images_shape.set_shape([3])
        inp_exps_tf.set_shape([3])
        ref_exps_tf.set_shape([3])
        return images_tf, images_shape, inp_exps_tf, ref_exps_tf

    def resize(image, hw):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(image, hw, align_corners=True)
        return image[0]

    def aug_images(images, images_shape, inp_exps, ref_exps):
        images = tf.reshape(images, images_shape)
        distortions = tf.random_uniform([3], 0, 1.0, dtype=tf.float32)
        if motion_data:
            images = tf.cond(tf.less(distortions[2], 0.25),
                             lambda: resize(images, config.train_hw),
                             lambda: images)
        images = tf.random_crop(images, [config.train_hw[0], config.train_hw[1], images_count * 3])
        if aug:
            images = tf.image.rot90(images, tf.cast(distortions[0] * 4, tf.int32))
            images = tf.cond(tf.less(distortions[1], 0.5), lambda: tf.image.flip_left_right(images), lambda: images)
        return images, inp_exps, ref_exps

    dataset = tf.data.Dataset.from_tensor_slices(scenes)
    dataset = dataset.map(scene2images, num_parallel_calls=config.BATCH_SIZE).cache()
    dataset = dataset.map(aug_images, num_parallel_calls=config.BATCH_SIZE)
    if shuffle:
        dataset = dataset.shuffle(64)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(config.BATCH_SIZE).prefetch(10)
    ims, _inp_exps, _ref_exps = dataset.make_one_shot_iterator().get_next()
    return ims, _inp_exps, _ref_exps


def _get_train_data(motion_data: bool, aug: bool, shuffle: bool, repeat: bool, mu=None):
    """
    :param motion_data:
    :param aug:
    :param shuffle:
    :param repeat:
    :return: (ldr1, ldr2, ldr3), (ldr1r, ldr2r, ldr3r), (tp1, tp2, tp3), (tp1r, tp2r, tp3r), hdr, inp_exps, ref_exps
    """
    im_fns = ['input_1.tif', 'input_2.tif', 'input_3.tif',
              'ref_1.tif', 'ref_2.tif', 'ref_3.tif',
              'ref_hdr.hdr']
    images_count = len(im_fns)

    ims, inp_exps, ref_exps = _get_scene_data(im_fns, motion_data, aug, shuffle, repeat)

    split_ims = tf.split(ims, images_count, axis=-1)
    ldr1, ldr2, ldr3, ldr1r, ldr2r, ldr3r, hdr = split_ims[:7]
    hdr1, hdr2, hdr3 = [hdr_utils.ldr2hdr(ldr, tf.reshape(inp_exps[..., ei], [-1, 1, 1, 1]))
                        for ldr, ei in zip([ldr1, ldr2, ldr3], range(3))]
    hdr1r, hdr2r, hdr3r = [hdr_utils.ldr2hdr(ldr, tf.reshape(ref_exps[..., ei], [-1, 1, 1, 1]))
                           for ldr, ei in zip([ldr1r, ldr2r, ldr3r], range(3))]
    tp1, tp2, tp3 = [hdr_utils.tonemap(hdr, mu=mu) for hdr in [hdr1, hdr2, hdr3]]
    tp1r, tp2r, tp3r = [hdr_utils.tonemap(hdr, mu=mu) for hdr in [hdr1r, hdr2r, hdr3r]]
    # tp1, tp2, tp3 = [hdr_utils.tonemap(hdr_utils.ldr2hdr(ldr, tf.reshape(inp_exps[..., ei], [-1, 1, 1, 1])), mu=mu)
    #                  for ldr, ei in zip([ldr1, ldr2, ldr3], range(3))]
    # tp1r, tp2r, tp3r = [hdr_utils.tonemap(hdr_utils.ldr2hdr(ldr, tf.reshape(ref_exps[..., ei], [-1, 1, 1, 1])), mu=mu)
    #                     for ldr, ei in zip([ldr1r, ldr2r, ldr3r], range(3))]
    return (ldr1, ldr2, ldr3), (ldr1r, ldr2r, ldr3r), \
           (tp1, tp2, tp3), (tp1r, tp2r, tp3r), \
           (hdr1, hdr2, hdr3), (hdr1r, hdr2r, hdr3r), hdr, inp_exps, ref_exps


@wrap_tf_name_scope()
def get_train_data(mu=None):
    return _get_train_data(False, True, True, True, mu=mu)


def _test_get_train_data():
    return _get_train_data(False, False, False, False)


@wrap_tf_name_scope()
def get_train_motion_data():
    return _get_train_data(True, True, True, True)


def _test_get_train_motion_data():
    return _get_train_data(True, False, False, False)


@wrap_tf_name_scope()
def get_tp_data():
    im_fns = ['ref_hdr.hdr', 'ref_hdr_Interior 3.tif']
    ims, inp_exps, ref_exps = _get_scene_data(im_fns, False, True, True, True)
    hdr, cus_tp = tf.split(ims, 2, axis=-1)
    return hdr, cus_tp


def make_flist(file_name):
    with open(file_name, 'r') as fr:
        res = fr.read().split('\n')
        res = [i for i in res if len(i) > 2]
    return res


@wrap_tf_name_scope()
def get_val_data(cache=True):
    pass

def _test():
    # res = _test_get_train_data()
    res = _test_get_train_motion_data()
    (ldr1, ldr2, ldr3), (ldr1r, ldr2r, ldr3r), (tp1, tp2, tp3), (tp1r, tp2r, tp3r), hdrs, hdrsr, \
    hdr, _inp_exps, _ref_exps = res
    ldr = tf.concat([ldr1, ldr2, ldr3], axis=2)
    ldr_ref = tf.concat([ldr1r, ldr2r, ldr3r], axis=2)
    tp = tf.concat([tp1, tp2, tp3], axis=2)
    tp_ref = tf.concat([tp1r, tp2r, tp3r], axis=2)
    hdr_tp = hdr_utils.tonemap(hdr)
    tk.summary.images(ldr, 3, 'ldr')
    tk.summary.images(ldr_ref, 3, 'ldr_ref')
    tk.summary.images(tp, 3, 'tp')
    tk.summary.images(tp_ref, 3, 'tp_ref')
    tk.summary.images(hdr_tp, 3, 'hdr_tp')

    summary_merged = tf.summary.merge_all()
    step = 0
    tic = tk.TimeTic()
    with tk.session() as sess:
        summary_writer = tf.summary.FileWriter('tmp/inp_' + tk.utils.get_time(), sess.graph)
        while step < config.TRAIN_SIZE:
            summary_writer.add_summary(sess.run(summary_merged), step)
            print('step: {}, tic: {}'.format(step, tic.tic()))
            step += 1


if __name__ == '__main__':
    _test()
