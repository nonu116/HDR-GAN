import os

import tensorflow as tf

import data
import hdr_utils
import tensorkit as tk
from config import config
from loss import tf_ms_ssim_loss
from model.gan import PatchDiscriminator
from model.unetpps import UnetppGeneratorS
from tensorkit import hparam
from tensorkit.gan_loss import sphere as sphere_gan
from tensorkit.utils import get_time
from utils import l1_loss

UnetppGenerator, UnetGeneratorS = None, None


def make_log_dir():
    if config.UNETPPS:
        assert config.GENERATOR in ['', 'unetpps']
        config.GENERATOR = 'unetpps'
    tag = config.GENERATOR
    if tag == '':
        tag = 'unetpps'
    p = '_'.join([get_time(), str(os.getpid()),
                  tag if config.UNETPPS else '',
                  config.safely_get('GAN', default=''),
                  'dpatchOri' if config.safely_get('PATCH_ORI', False) else '',
                  config.NORM,
                  'mu{}'.format(config.MU) if config.MU is not None and config.MU != 5000. else '',
                  'inHDR' if config.IN_HDR else '',
                  'oHDR' if config.OUT_HDR else '',
                  'lsTP' if not config.N_LOSS_TP else '',
                  'lsHDR' if config.LOSS_HDR else '',
                  'rp' if config.random_place else '',
                  config.safely_get('TAG', ''),
                  ]).strip('_')
    while '__' in p:
        p = p.replace('__', '_')
    log_dir = os.path.join(config.LOG_DIR, p)
    return log_dir, tag


HPARAM_FILE = 'c{}.conf'.format(os.getpid())


def gradient_sum(loss, var, tag):
    assert len(var) > 0
    grad = tf.gradients(loss, var)
    for g, v in zip(grad, var):
        if g is not None:
            tk.summary.histogram(g, '{}/{}'.format(tag, v.name))


def _graph(datas, model, discriminator, train, reuse, summary_feat, mu=None):
    (ldr1, ldr2, ldr3), (ldr1r, ldr2r, ldr3r), (tp1, tp2, tp3), (tp1r, tp2r, tp3r), \
    (hdr1, hdr2, hdr3), (hdr1r, hdr2r, hdr3r), hdr, _inp_exps, _ref_exps = datas
    with tf.name_scope('inputs'):
        if config.random_place:
            random = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
            ldr2, tp2, hdr2 = tf.cond(tf.less(random[0], 0.25),
                                      lambda: tf.cond(tf.less(random[1], 0.3), lambda: (ldr1r, tp1r, hdr1r),
                                                      lambda: (ldr3r, tp3r, hdr3r)),
                                      lambda: (ldr2, tp2, hdr2))
        image1 = tf.concat([ldr1, hdr1 if config.IN_HDR else tp1], axis=-1)
        image2 = tf.concat([ldr2, hdr2 if config.IN_HDR else tp2], axis=-1)
        image3 = tf.concat([ldr3, hdr3 if config.IN_HDR else tp3], axis=-1)
        gt_hdr = hdr
        gt_tonemap = hdr_utils.tonemap(hdr, mu=mu)

    outputs, features = model.graph(image1, image2, image3,
                                    train=train, summary_feat=summary_feat, get_features=True, reuse=reuse)
    if config.OUT_HDR:
        out_hdrs = outputs
        out_tps = [hdr_utils.tonemap(h, mu=mu) for h in out_hdrs]
    else:
        out_tps = outputs
        out_hdrs = [hdr_utils.itonemap(tp, mu=mu) for tp in out_tps]

    with tf.name_scope('gan_real'):
        gan_real = tf.identity(gt_tonemap)

    dis_real = discriminator.graph(gan_real, reuse=reuse) if discriminator is not None else None
    dis_fakes = []
    losses = dict()
    sums = dict()
    d_losses, g_losses = dict(), dict()
    # ==================== alpha =======================
    l1_hdr_alpha, d_alpha, g_alpha = None, None, None
    with tf.variable_scope('alpha', reuse=tf.AUTO_REUSE):
        if True or config.LOSS_HDR:
            l1_hdr_alpha = hparam.get_hparamter('l1_hdr_alpha', initializer=1.)
            hparam.set_sync(l1_hdr_alpha, 'L1_HDR_ALPHA', HPARAM_FILE)
            sums['alpha/l1_hdr_alpha'] = l1_hdr_alpha
        if discriminator is not None:
            g_alpha = hparam.get_hparamter('g_alpha', initializer=1.)
            d_alpha = hparam.get_hparamter('d_alpha', initializer=1.)
            hparam.set_sync(g_alpha, 'G_ALPHA', HPARAM_FILE)
            hparam.set_sync(d_alpha, 'D_ALPHA', HPARAM_FILE)
            sums['alpha/g_alpha'] = g_alpha
            sums['alpha/d_alpha'] = d_alpha

    # ================== loss item =====================
    for ind, (fake_hdr, fake_tp) in enumerate(zip(out_hdrs, out_tps)):
        if not config.N_LOSS_TP:
            losses['l1/{}'.format(ind)] = l1_loss(gt_tonemap, fake_tp)
        else:
            sums['l1/{}'.format(ind)] = l1_loss(gt_tonemap, fake_tp)
        if config.LOSS_HDR:
            losses['l1_hdr/{}'.format(ind)] = l1_loss(gt_hdr, fake_hdr) * l1_hdr_alpha
        else:
            sums['l1_hdr/{}'.format(ind)] = l1_loss(gt_hdr, fake_hdr) * l1_hdr_alpha
        if config.MSSIM_L1:
            sums['l1/{}'.format(ind)] = l1_loss(gt_tonemap, fake_tp)
            sums['MSSIM/{}'.format(ind)] = tf_ms_ssim_loss(gt_tonemap, fake_tp)
            _alpha = 0.84
            losses['MSSIM_L1/{}'.format(ind)] = (1 - _alpha) * sums['l1/{}'.format(ind)] \
                                                + _alpha * sums['MSSIM/{}'.format(ind)]

        if discriminator is not None:
            gan_fake = fake_tp
            if config.GAN == 'sphere':
                dis_fake = discriminator.graph(gan_fake, reuse=True)
                dis_fakes.append(dis_fake)
                g_loss, d_loss, (distance_real, distance_fake, g_convergence_to_zero, d_convergence_to_min) = \
                    sphere_gan(dis_real, dis_fake, None, 3, reuse=ind != 0)
                g_losses['g_loss/{}'.format(ind)] = g_loss
                d_losses['d_loss/{}'.format(ind)] = d_loss
                sums['sphere/distance_real/{}'.format(ind)] = distance_real
                sums['sphere/distance_fake/{}'.format(ind)] = distance_fake
                sums['sphere/g_convergence_to_zero/{}'.format(ind)] = g_convergence_to_zero
                sums['sphere/d_convergence_to_min/{}'.format(ind)] = d_convergence_to_min
            elif config.GAN == 'pgan':
                dis_real = discriminator.graph(gan_real, reuse=reuse or ind != 0)
                dis_fake = discriminator.graph(gan_fake, reuse=True)
                distance = l1_loss(dis_real, dis_fake)
                g_losses['g_loss/{}'.format(ind)] = distance
                d_losses['d_loss/{}'.format(ind)] = -distance
                sums['gan/g_loss/{}'.format(ind)] = distance
                sums['gan/d_loss/{}'.format(ind)] = -distance
            else:
                raise RuntimeError('invalid gan: {}'.format(config.GAN))

    # ================== loss total ====================
    tk.logger.info('loss key: {}'.format(' '.join(losses.keys())))
    _loss = loss = tf.add_n(list(losses.values()))
    g_loss, d_loss = None, None
    if discriminator is not None:
        g_loss, d_loss = tf.add_n(list(g_losses.values())) * g_alpha, tf.add_n(list(d_losses.values())) * d_alpha
        loss += g_loss

    # ==================== summary =====================
    assert all(k not in sums for k in losses.keys()), 'losses: {}\nsums:{}'.format(losses.keys(), sums.keys())
    sums.update(losses)
    for k, v in sums.items():
        tk.summary.scalar(v, k)
    with tf.name_scope('gradient_sum'):
        outvars = model.variable_out()
        # -------- gradient gan ---------
        if discriminator is not None:
            _outvars = discriminator.variable_out()
            with tf.name_scope('gan'):
                gradient_sum(g_loss, outvars, 'g_loss')
                gradient_sum(_loss, outvars, 'ng_loss')
                gradient_sum(d_loss, _outvars, 'd_loss')
        # -------- gradient l1 ----------
        _loss = [v for k, v in losses.items() if 'l1/' in k]
        if not config.N_LOSS_TP:
            with tf.name_scope('l1'):
                assert len(_loss) in [0, 4, config.DEPTH - 1], _loss
                if len(_loss) != 0:
                    _loss = tf.add_n(_loss)
                    gradient_sum(_loss, outvars, 'l1')
        # -------- gradient l1_hdr ----------
        if config.LOSS_HDR:
            with tf.name_scope('l1_hdr'):
                _loss = [v for k, v in losses.items() if 'l1_hdr/' in k]
                assert len(_loss) in [0, 4, config.DEPTH - 1], _loss
                if len(_loss) != 0:
                    _loss = tf.add_n(_loss)
                    gradient_sum(_loss, outvars, 'l1_hdr')
    # -------- images ----------
    if discriminator is not None:
        with tf.name_scope('sum_sphere'):
            gan_sum_real = tk.image.colorize(dis_real)
            gan_sum_real = tf.concat([gan_sum_real for _ in dis_fakes], 2)
            gan_sum_fake = tf.concat([i for i in dis_fakes], 2)
            gan_sum_fake = tk.image.colorize(gan_sum_fake)
            gan_sum_diff = tf.concat([tf.abs(i - dis_real) for i in dis_fakes], 2)
            gan_sum_diff = tk.image.colorize(gan_sum_diff)
            sum_im = tf.concat([gan_sum_real, gan_sum_fake, gan_sum_diff], axis=1)
        tk.summary.images(sum_im, 3, 'sphere')

    with tf.name_scope('summary'):
        reals = tf.concat([tp1, tp2, tp3, gt_tonemap], axis=2)
        fakes = out_tps + [tf.ones_like(out_tps[0]) for _ in range(4 - len(out_tps))]
        fakes = tf.concat(fakes, axis=2)
        sum_ims = tf.concat([reals, fakes], axis=1)
        tk.summary.images(sum_ims, 3, 'reals_fakes')

    return loss, d_loss


def graph():
    if config.UNETPPS:
        assert config.GENERATOR in ['', 'unetpps']
        config.GENERATOR = 'unetpps'
    if config.GENERATOR == 'unetpps':
        model = UnetppGeneratorS(depth=config.DEPTH, norm=config.NORM)
    elif config.GENERATOR == 'unets':
        model = UnetGeneratorS(depth=config.DEPTH, norm=config.NORM)
    elif config.GENERATOR in ['', 'unetpp']:
        model = UnetppGenerator(depth=config.DEPTH, norm=config.NORM)
    else:
        raise NotImplementedError('generator: {}'.format(config.GENERATOR))
    discriminator = PatchDiscriminator(ori=config.PATCH_ORI) if config.LOSS_GAN else None
    train_data = data.get_train_data(mu=config.MU)
    train_loss, d_loss = _graph(train_data, model, discriminator, train=True, reuse=None, summary_feat=True,
                                mu=config.MU)

    val_loss = None
    if config.VAL:
        val_data = data.get_val_data()
        with tf.name_scope('val'):
            val_loss, _ = _graph(val_data, model, None, train=False, reuse=True, summary_feat=False, mu=None)
    if config.LOSS_GAN:
        return (train_loss, d_loss), (model, discriminator), val_loss
    else:
        return train_loss, model, val_loss


def args(parser):
    parser.add_argument('--depth', dest='DEPTH', default=3, choices=(2, 3, 4, 5), type=int)
    parser.add_argument('--norm', '-norm', help='normalization',
                        dest='NORM', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
    parser.add_argument('--loss_gan', dest='LOSS_GAN', default=False, action='store_true')
    parser.add_argument('--gan', dest='GAN', default='sphere', choices=['sphere', 'wgan_gp', 'pgan'])
    parser.add_argument('--loss_hdr', dest='LOSS_HDR', default=False, action='store_true')
    parser.add_argument('--mssim_l1', dest='MSSIM_L1', default=False, action='store_true')
    parser.add_argument('--train_hw', dest='train_hw', type=int, nargs=2, default=...)
    parser.add_argument('--random_place', dest='random_place', default=False, action='store_true')
    parser.add_argument('--patch_ori', dest='PATCH_ORI', default=False, action='store_true')
    parser.add_argument('--mu', dest='MU', default=None, type=float)
    parser.add_argument('--in_hdr', dest='IN_HDR', default=False, action='store_true')
    parser.add_argument('--out_hdr', dest='OUT_HDR', default=False, action='store_true')
    parser.add_argument('--n_loss_tp', dest='N_LOSS_TP', default=False, action='store_true')
    parser.add_argument('--unetpps', dest='UNETPPS', default=False, action='store_true')
    parser.add_argument('--gen', dest='GENERATOR', default='', choices=('unetpps', 'unetpp', 'unets', 'unet'))
    return parser
