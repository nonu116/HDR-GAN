import cv2
import numpy as np
import tensorflow as tf

GAMMA = 2.2  # LDR and HDR domain transform parameter
MU = 5000.  # tonemapping parameter


def write_hdr(out_path, image):
    if len(image.shape) == 4:
        assert image.shape[0] == 1, 'invalid shape: {}'.format(image)
        image = image[0]
    assert len(image.shape) == 3 and image.shape[-1] == 3
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def read_hdr(hdr_path):  # output -1~1
    im = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return (im * 2. - 1.)[..., ::-1]


def hdr2ldr(hdr, expo):  # input/output -1~1
    return (tf.clip_by_value(((hdr + 1) / 2. * expo), 0, 1) ** (1 / GAMMA)) * 2. - 1


def ldr2hdr(ldr, expo):  # input/output -1~1
    return (((ldr + 1.) / 2.) ** GAMMA / expo) * 2. - 1


def tonemap_np(hdr, mu=MU):  # input/output -1~1
    if mu is None:
        mu = MU
    return np.log(1 + mu * (hdr + 1.) / 2.) / np.log(1 + mu) * 2. - 1


def itonemap_np(tp, mu=MU):
    if mu is None:
        mu = MU
    return ((1. + mu) ** ((tp + 1.) / 2) - 1) / mu * 2 - 1


def tonemap(hdr, mu=MU, name='tonemap'):  # input/output -1~1
    if mu is None:
        mu = MU
    with tf.name_scope(name):
        return tf.log(1 + mu * (hdr + 1.) / 2.) / tf.log(1 + mu) * 2. - 1


def itonemap(tp, mu=MU, name='itonemap'):
    if mu is None:
        mu = MU
    with tf.name_scope(name):
        return (tf.pow((1. + mu), ((tp + 1.) / 2)) - 1) / mu * 2 - 1
