import numpy as np
import tensorflow as tf

from tensorkit.model import BaseModel


def gan_qp(real, fake, discriminator: BaseModel, reuse=None):
    """
    https://kexue.fm/archives/6163
    :param real:
    :param fake:
    :param discriminator:
    :param reuse:
    :return: (g_loss, d_loss)
    """
    with tf.name_scope('gan_qp'):
        dis_pos = discriminator.graph(real, reuse=reuse)
        dis_neg = discriminator.graph(fake, reuse=True)
        d_loss = dis_pos - dis_neg
        distance = 10 * (tf.reduce_mean(tf.abs(real - fake), axis=[1, 2, 3]) + 1e-8)
        d_loss = tf.reduce_mean(-d_loss + 0.5 * tf.multiply(d_loss, d_loss) / distance, name='d_loss')
        g_loss = tf.reduce_mean(dis_pos - dis_neg, name='g_loss')
        return g_loss, d_loss


def _cal_wgan(dis_pos, dis_neg):
    """
    :param dis_pos: [b, 1]
    :param dis_neg: [b, 1]
    :return: g_loss, d_loss, k_w_distance
    """
    dis_pos_mean = tf.reduce_mean(dis_pos)
    dis_neg_mean = tf.reduce_mean(dis_neg)
    with tf.name_scope('d_loss'):
        d_loss = -dis_pos_mean + dis_neg_mean
    with tf.name_scope('g_loss'):
        g_loss = -dis_neg_mean
    with tf.name_scope('k_w_distance'):
        k_w_distance = dis_pos_mean - dis_neg_mean
    return g_loss, d_loss, k_w_distance


def wgan(real, fake, discriminator: BaseModel, reuse=None):
    """
    :param real:
    :param fake:
    :param discriminator:
    :param reuse:
    :return: g_loss, d_loss, k_w_distance
    """
    with tf.name_scope('wgan'):
        dis_pos = discriminator.graph(real, reuse=reuse)
        dis_neg = discriminator.graph(fake, reuse=True)
        return _cal_wgan(dis_pos, dis_neg)


def wgan_gp(real, fake, discriminator: BaseModel, gp_alpha=1., reuse=None):
    """
    :param real:
    :param fake:
    :param discriminator:
    :param gp_alpha:
    :param reuse:
    :return: g_loss, d_loss, (k_w_distance, grad_penalty)
    """
    with tf.name_scope('wgan_gp'):
        batch_size = tf.shape(real)[0]
        x_dim = len(real.get_shape().as_list()) - 1
        alpha = tf.random_uniform(shape=[batch_size] + [1 for _ in range(x_dim)], minval=0., maxval=1.)
        interpolated = alpha * real + (1. - alpha) * fake

        dis_pos = discriminator.graph(real, reuse=reuse)
        dis_neg = discriminator.graph(fake, reuse=True)
        dis_inter = discriminator.graph(interpolated, reuse=True)

        grad = tf.gradients(dis_inter, [interpolated])[0]
        grad_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[i + 1 for i in range(x_dim)]) + 1e-8)
        grad_penalty = tf.reduce_mean((grad_l2_norm - 1.) ** 2, name='grad_penalty')

        g_loss, d_loss, k_w_distance = _cal_wgan(dis_pos, dis_neg)
        d_loss = d_loss + gp_alpha * grad_penalty
        return g_loss, d_loss, k_w_distance, grad_penalty


class SphereGAN(object):
    @staticmethod
    def stereo_graphic_project(u):
        shape_ori = u.get_shape().as_list()
        if len(shape_ori) != 2:  # flatten
            dims = np.prod(shape_ori[1:])
            u = tf.reshape(u, (-1, dims))
        u_norm2 = tf.pow(tf.norm(u, axis=1), 2)
        p = tf.divide(2 * tf.transpose(u), u_norm2 + 1)
        tmp = tf.divide((u_norm2 - 1), (u_norm2 + 1))
        p = tf.concat([p, [tmp]], axis=0)
        p = tf.transpose(p)
        return p

    @staticmethod
    def reference_point(point):
        assert len(point.get_shape().as_list()) == 2
        dim = point.get_shape().as_list()[-1]
        ref_p_np = np.zeros((1, dim)).astype(np.float32)
        ref_p_np[0, 0] = 1.0
        return tf.constant(ref_p_np)

    @staticmethod
    def _dist_sphere(a, b, r):
        return tf.acos(tf.matmul(a, tf.transpose(b))) ** r

    @staticmethod
    def dist_weight_mode(r, weight_mode, decay_ratio):
        if weight_mode == 'normalization':
            decayed_dist = (np.pi / decay_ratio) ** r
        elif weight_mode == 'half':
            decayed_dist = np.pi ** r
        else:
            decayed_dist = 1
        return decayed_dist

    @staticmethod
    def make_loss(real, fake, discriminator: BaseModel = None,
                  moments: int = 3, weight_mode=None, decay_ratio=3, reuse=None):
        """
        :param real:
        :param fake:
        :param discriminator:
        :param moments: [3] is suggested but [1] is enough.
        :param weight_mode: ['normalization', 'half', else]
        :param decay_ratio: for normalization weight_mode
        :param reuse
        :return: g_loss, d_loss, (distance_real, distance_fake, g_convergence_to_zero, d_convergence_to_min)
        """
        with tf.name_scope('sphere_gan'):
            if discriminator is not None:
                dis_real = discriminator.graph(real, reuse=reuse)
                dis_fake = discriminator.graph(fake, reuse=True)
            else:
                dis_real = real
                dis_fake = fake
            dis_real_sphere = SphereGAN.stereo_graphic_project(dis_real)
            dis_fake_sphere = SphereGAN.stereo_graphic_project(dis_fake)
            ref_sphere = SphereGAN.reference_point(dis_fake_sphere)

            distance_fake = []
            distance_real = []
            for _r in range(moments):
                r = _r + 1
                df = SphereGAN._dist_sphere(dis_fake_sphere, ref_sphere, r)
                dr = SphereGAN._dist_sphere(dis_real_sphere, ref_sphere, r)
                distance_fake.append(tf.reduce_mean(df) / SphereGAN.dist_weight_mode(r, weight_mode, decay_ratio))
                distance_real.append(tf.reduce_mean(dr) / SphereGAN.dist_weight_mode(r, weight_mode, decay_ratio))

            distance_fake = tf.add_n(distance_fake, name='distance_fake')
            distance_real = tf.add_n(distance_real, name='distance_real')
            g_loss = -distance_fake
            d_loss = -distance_real + distance_fake
            d_convergence_to_min = d_loss
            g_convergence_to_zero = d_loss

            return g_loss, d_loss, (distance_real, distance_fake, g_convergence_to_zero, d_convergence_to_min)


sphere = SphereGAN.make_loss
