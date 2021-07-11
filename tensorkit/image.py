import cv2
import numpy as np
import tensorflow as tf

from tensorkit.log import logger, Color
from tensorkit.sess import default_session

try:
    _cmap = cv2.COLORMAP_VIRIDIS
except:
    _cmap = cv2.COLORMAP_JET


def colorize_np(gray_image, value_min=None, value_max=None, cmap=_cmap, normalize=True):
    if isinstance(cmap, str):
        cmap = getattr(cv2, 'COLORMAP_{}'.format(cmap.upper()))
    assert isinstance(gray_image, np.ndarray)
    inp_shape = gray_image.shape
    assert len(inp_shape) in [3, 4] and inp_shape[-1] == 1, 'input shape: %s' % inp_shape
    if len(inp_shape) == 3:
        gray_image = np.expand_dims(gray_image, 0)  # [b, h, w, 1]
    # normalize
    value_min = np.min(gray_image, axis=(1, 2), keepdims=True) if value_min is None else value_min
    value_max = np.max(gray_image, axis=(1, 2), keepdims=True) if value_max is None else value_max
    if normalize:
        gray_image = (gray_image - value_min) / (value_max - value_min)
    else:
        if np.min(value_min) < 0 or np.max(value_max) > 1:
            logger.info(Color.yellow(
                'gray_image are not normalize whose value exceed [0, 1]: [{}, {}]'.format(np.min(value_min),
                                                                                          np.max(value_max)), True))
    gray_image = (gray_image * 255.).round().astype(np.uint8)
    res = [cv2.applyColorMap(gray_image[i], cmap) for i in range(gray_image.shape[0])]
    res = np.stack(res, 0)[..., ::-1]  # BGR2RGB
    return res / 127.5 - 1.


def colorize_tf(gray_image, value_min=None, value_max=None, cmap=_cmap, normalize=True, name=None):
    with tf.name_scope('heatmap' if name is None else name):
        if isinstance(cmap, str):
            cmap = getattr(cv2, 'COLORMAP_{}'.format(cmap.upper()))

        inp_shape = gray_image.get_shape().as_list()
        assert len(inp_shape) in [3, 4] and inp_shape[-1] == 1, 'input shape: %s' % inp_shape
        if len(inp_shape) == 3:
            gray_image = tf.expand_dims(gray_image, 0)
        gray_image = tf.squeeze(gray_image, axis=-1)  # [b, h, w]

        # normalize
        if normalize:
            value_min = tf.reduce_min(gray_image, axis=[1, 2], keepdims=True) if value_min is None else value_min
            value_max = tf.reduce_max(gray_image, axis=[1, 2], keepdims=True) if value_max is None else value_max
            gray_image = (gray_image - value_min) / (value_max - value_min)

        # quantize
        indices = tf.cast(tf.round(gray_image * 255.), tf.int32)

        # gather
        colors = np.linspace(0, 255, 256).reshape([1, -1, 1]).astype(np.uint8)
        colors = cv2.applyColorMap(colors, cmap)[..., ::-1]  # BGR2RGB
        colors = colors.reshape(-1, 3) / 127.5 - 1.
        colors = tf.constant(colors, dtype=tf.float32)
        value = tf.gather(colors, indices)
        return value


def colorize(gray_image, value_min=None, value_max=None, cmap=_cmap, normalize=True):
    """
    Mapping grayscale images to heat_map images
    By default it will normalize the input value to the range [0..1] before mapping
    to a grayscale colormap.
    Arguments:
      - gray_image: Tensor or Numpy Array of shape [batch, height, width, 1] or [height, width, 1].
      - value_min: the minimum value of the range used for normalization.
        (Default: value minimum)
      - value_max: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: cv2.COLORMAP_JET or 'jet'
        (Default: cv2.COLORMAP_JET)
      - normalize: whether normalizing the input value to the range 0..1 before mapping
        to a grayscale colormap. Make sure input value is in range 0..1 if False
    Example usage:
    ```
    output = tf.random_uniform(shape=[10, 256, 256, 1])
    output_color = colorize(output, value_min=0.0, value_max=1.0, cmap='jet')
    assert output_color.get_shape().as_list() == [10, 256, 256, 3]
    ```

    Returns a 4D tensor of shape [batch, height, width, 3].
    """
    if isinstance(gray_image, np.ndarray):
        return colorize_np(gray_image, value_min, value_max, cmap, normalize)
    else:
        return colorize_tf(gray_image, value_min, value_max, cmap, normalize)


def zoom_image_np(image, height_min, width_min, interpolation=cv2.INTER_CUBIC):
    """
    scale image proportionally to make sure image height and width greater to height_min and width_min respectively
    :param image: image with shape [height, width, channel] or [height, width]
    :param height_min:
    :param width_min:
    :param interpolation:
    :return: image with shape [h, w, channel]
    """
    assert len(image.shape) in (2, 3), 'invalid image with shape: {}'.format(image.shape)
    height_ori, width_ori = image.shape[:2]
    hr = 1. * height_min / height_ori
    wr = 1. * width_min / width_ori
    if np.max([hr, wr]) > 1.:
        r = np.max([hr, wr])
        h = np.ceil(r * height_ori)
        w = np.ceil(r * width_ori)
        image = cv2.resize(image, dsize=(int(w), int(h)), interpolation=interpolation)
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
    return image


def read_image(file_name, color_format='RGB', batch_shape=None):
    im_np = read_image_np(file_name, color_format, batch_shape)
    return tf.constant(im_np)


def read_image_np(file_name, color_format='RGB', batch_shape=None):
    """
    read image and normalize
    :param file_name:
    :param color_format:
    :return: rgb image numpy in value [-1., 1.] or gray image in [0., 1.],
             shape is [height, width, channel] or [batch_shape, height, width, channel] if
             batch_shape is not None.
    """
    assert color_format in ['RGB', 'BGR', 'GRAY', 'GREY']
    if color_format in ['GRAY', 'GREY']:
        im = cv2.imread(file_name, flags=cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        im = im / 255.
    else:
        im = cv2.imread(file_name, flags=cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(file_name)
        im = im / 127.5 - 1.
        im = im if color_format == 'BGR' else im[..., ::-1]
    if batch_shape is not None and batch_shape > 0:
        im = np.tile(np.expand_dims(im, 0), [batch_shape, 1, 1, 1])
    return im.astype(np.float32)


def save_image(image, file_name, color_format='RGB'):
    """
    :param image: image tensor with shape [1, height, width, channel] or [height, width, channel]
    :param file_name:
    :param color_format: ['RGB', 'BGR', 'GRAY', 'GREY']
    :return:
    """
    if not isinstance(image, np.ndarray):
        sess = default_session(-1)
        image = sess.run(image)
    return _save_image_np(image, file_name, color_format)


def _save_image_np(image, file_name, color_format='RGB'):
    """
    :param image: image numpy with shape [1, height, width, channel] or [height, width, channel]
    :param file_name:
    :param color_format: ['RGB', 'BGR', 'GRAY', 'GREY']
    :return:
    """
    image_shape = image.shape
    color_format = color_format.upper()
    assert len(image_shape) == 3 or (len(image_shape) == 4 and image_shape[0] == 1), 'Invalid image_shape: {}'.format(
        image_shape)
    if color_format in ['RGB', 'BGR']:
        assert image_shape[-1] == 3, 'color image channel must be 3, image_shape: {}'.format(image_shape)
    elif color_format in ['GRAY', 'GREY']:
        assert image_shape[-1] == 1, 'gray image channel must be 1, image_shape: {}'.format(image_shape)
    else:
        raise RuntimeError('Unsupported color_format: %s' % color_format)

    if len(image_shape) == 4:
        image = image[0]

    if color_format in ['RGB', 'BGR']:
        value_min, value_max = -1., 1.
    else:
        value_min, value_max = 0., 1.

    if not (value_min <= np.min(image) and np.max(image) <= value_max):
        logger.warning(Color.yellow('Image value is out of range: [{}, {}]'.format(
            np.min(image), np.max(image)), bold=True))
        image = np.clip(image, value_min, value_max)

    image = (image - value_min) * 255. / (value_max - value_min)
    if color_format == 'RGB':
        image = image[..., ::-1]
    return cv2.imwrite(file_name, image)
