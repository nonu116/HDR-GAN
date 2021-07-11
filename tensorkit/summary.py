import os

import tensorflow as tf

from tensorkit.image import colorize

_merge_key = None


def set_default_collections(collections):
    global _merge_key
    _merge_key = collections


def scalar(tensor, name, collections=None):
    with tf.device('/cpu:0'):
        key = collections if collections is not None else _merge_key
        tf.summary.scalar(name.replace(':', '_'), tensor, collections=key)


def histogram(values, name, collections=None):
    with tf.device('/cpu:0'):
        key = collections if collections is not None else _merge_key
        tf.summary.histogram(name.replace(':', '_'), values, collections=key)


def images(tensor, max_outputs, name, color_format='RGB', collections=None):
    with tf.name_scope('tensor_to_image'):
        if color_format == 'BGR':
            img = tf.clip_by_value(
                (tf.reverse(tensor, [-1]) + 1.) * 127.5, 0., 255.)
        elif color_format == 'RGB':
            img = tf.clip_by_value((tensor + 1.) * 127.5, 0, 255)
        elif color_format == 'GREY':
            img = tf.clip_by_value(tensor * 255., 0, 255)
        elif color_format == 'HEAT_MAP':
            img = colorize(tensor, value_min=0., value_max=1., cmap='jet', normalize=False)
            img = tf.clip_by_value((img + 1.) * 127.5, 0, 255)
        else:
            raise NotImplementedError("color format is not supported.")
    with tf.device('/cpu:0'):
        key = collections if collections is not None else _merge_key
        tf.summary.image(name.replace(':', '_'), img, max_outputs=max_outputs, collections=key)


def get_tags_from_event_file(event_path):
    tags = set()
    for step, event in enumerate(tf.train.summary_iterator(event_path)):
        for value in event.summary.value:
            tags.add(value.tag)
    return tags


def smaller_event_file(event_path, freq_dic, out_path):
    """
    :param event_path:
    :param freq_dic: {'loss': 3} , smaller summary of loss to 1/3, 'loss' is a pattern of summary_name
    :param out_path
    :return:
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    writer = tf.summary.FileWriter(out_path)
    try:
        for step, event in enumerate(tf.train.summary_iterator(event_path)):
            print('\rprocess: {} ...'.format(step), end='')
            event_type = event.WhichOneof('what')
            if event_type != 'summary':
                writer.add_event(event)
            else:
                wall_time = event.wall_time
                filtered_values = []
                for value in event.summary.value:
                    patterns = value.tag.split('/')
                    freq = -1
                    for p in patterns:
                        if p in freq_dic.keys():
                            freq = freq_dic[p]
                    if freq <= 0 or (freq > 0 and int(step % freq) == 0):
                        filtered_values.append(value)

                summary = tf.Summary(value=filtered_values)
                filtered_event = tf.summary.Event(summary=summary,
                                                  wall_time=wall_time,
                                                  step=event.step)
                writer.add_event(filtered_event)
    except Exception as e:
        print('ERROR:', e)
        print()
    writer.close()
