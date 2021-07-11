import argparse
import importlib
import os
import sys

import tensorflow as tf

import tensorkit as tk
import train_unetpps as train_module
from config import config
from tensorkit import TimeTic
from tensorkit import Trainer
from tensorkit import hparam
from tensorkit.restore import Restore
from tensorkit.save import Saver
from tensorkit.utils import BackupFiles
from utils import optimize_op
from utils import parse_args


def _make_backup():
    bf = BackupFiles()
    bf.include_dirs('.', suffix='.py', recursive=False) \
        .include_dirs(*config.BACKUP_DIR, suffix='.py', recursive=True)
    return bf


def mk_lr(lr_init):
    lr = hparam.get_hparamter('lr', initializer=lr_init)
    hparam.set_sync(lr, 'LR', 'c{}.conf'.format(os.getpid()))
    lr_init_op = tf.assign(lr, lr_init)
    decay_rate = 0.995
    decay_op = tf.assign(lr, tf.maximum(lr * decay_rate, 1e-6))
    return lr, lr_init_op, decay_op


def train():
    sess = None
    if not config.safely_get('CHECK_VARIABLES_NUM', False):
        sess = tk.session(config.CUDA_VISIBLE_DEVICES, config.ALLOW_GROWTH)

    bf = _make_backup()

    loss, model, val_loss = train_module.graph()

    d_loss, discriminator, train_op_d = None, None, None
    if type(loss) in [tuple, list]:
        loss, d_loss = loss
        model, discriminator = model

    if config.safely_get('CHECK_VARIABLES_NUM', False):
        model.view_trainable_variables()
        exit()

    lr, lr_init_op, lr_decay_op = mk_lr(config.LR)
    tk.summary.scalar(lr, 'learning_rate')
    if config.OPT == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif config.OPT == 'sgd_m':
        optimizer = tf.train.MomentumOptimizer(lr, momentum=config.MOMENTUM)
    elif config.OPT == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    else:
        raise RuntimeError('invalid optimizer: {}'.format(config.OPT))
    if config.OPT == 'sgd':
        optimizer_d = tf.train.GradientDescentOptimizer(lr)
    else:
        optimizer_d = tf.train.AdamOptimizer(lr)

    var_list = model.trainable_variables()
    assert len(var_list) > 0

    tk.logger.info(' train variable '.center(100, '='))
    tk.logger.info('\n'.join([i.name for i in var_list]))
    tk.logger.info('=' * 100)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op, grads_and_vars = optimize_op(loss, var_list, optimizer, 1.)
        if d_loss is not None:
            train_op_d, grads_and_vars_d = optimize_op(d_loss, discriminator.trainable_variables(), optimizer_d, 1.)

    log_dir, tag = train_module.make_log_dir()

    trainer = Trainer() \
        .set_epoch(config.BATCH_SIZE, config.TRAIN_SIZE, config.EPOCH) \
        .set_session(sess) \
        .add_train_op([train_op, loss]) \
        .set_summary(log_dir, config.SUMMARY_STEP, val_loss) \
        .set_saver(tf.global_variables(), log_dir, 'model', config.SAVE_STEP, max_to_keep=3) \
        .add_prepare_listener(lambda _: model.view_trainable_variables()) \
        .add_prepare_listener(lambda _: bf.run(os.path.join(log_dir, 'source'))) \
        .add_prepare_listener(lambda _: config.dump(os.path.join(log_dir, 'config.yml'))) \
        .finalize_graph()

    tic = TimeTic()
    saver = Saver(save_dir=log_dir, model_name='best_model', var_list=tf.global_variables(), max_to_keep=3)
    if config.ABLATION:
        abla_saver = Saver(save_dir=log_dir, model_name='abla_model', var_list=tf.global_variables(), max_to_keep=100)
    min_vl = 1e5

    def progress(sess, res, val_loss_np, step_info):
        _, loss_np = res
        print('\r {}, {}, loss: {:.6f}, tic: {:.4f}'.format(tag, step_info, loss_np, tic.tic()), end='')
        if config.ABLATION:
            if step_info.current_iter == 251600 - 10:
                with open('c{}.conf'.format(os.getpid()), 'w') as fw:
                    fw.write('LR: 1e-5')
        if step_info.is_per(global_iter_per=1000):
            vl = 0
            for i in range(220):
                vl += sess.run(val_loss)
            nonlocal min_vl
            if vl <= min_vl:
                saver.call(sess, step_info.current_iter)
                min_vl = vl
        if step_info.is_per(global_iter_per=200):
            hparam.sync_hparamters(sess)
            if config.LR_DECAY:
                sess.run(lr_decay_op)
        if config.ABLATION:
            if step_info.current_iter in [80000, 251000, 328000, 501000]:
                abla_saver.call(sess, step_info.current_iter)
        if train_op_d is not None:
            sess.run(train_op_d)

    trainer.on_train_iter_res(progress)

    def restore(sess):
        for cf in config.CKPT_FILE:
            Restore().init(ckpt_dir=log_dir, ckpt_file=cf, optimistic=True).restore(sess)
        vgg_vars = [v for v in tf.contrib.framework.get_variables_to_restore() if 'vgg' in v.name]
        if len(vgg_vars) > 0:
            Restore().init(var_list=vgg_vars, ckpt_file=config.VGG_CKPT, optimistic=True).restore(sess)

    def variables2loss(gv, ls):
        content = [' update by loss: {} '.format(ls).center(100, '=')]
        content += [i.name for _, i in gv]
        content += ['=' * 100]
        tk.logger.info('\n'.join(content))

    trainer.add_prepare_listener(restore)
    trainer.add_prepare_listener(lambda sess: sess.run(lr_init_op))
    trainer.add_prepare_listener(lambda _: variables2loss(grads_and_vars, loss))
    if train_op_d is not None:
        trainer.add_prepare_listener(lambda _: variables2loss(grads_and_vars_d, d_loss))
    trainer.add_prepare_listener(lambda _: tk.logger.info('CMD: {}'.format(' '.join(sys.argv))))
    tk.logging_to_file(os.path.join(log_dir, 'log.txt'), with_color=False)
    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', dest='ABLATION', default=False, action='store_true')
    parser.add_argument('--epoch', dest='EPOCH', default=..., type=int)
    parser.add_argument('--tag', dest='TAG', default=...)
    parser.add_argument('--gpu', dest='CUDA_VISIBLE_DEVICES', default=..., type=int)
    parser.add_argument('--ckpt', dest='CKPT_FILE', default=tuple(), nargs='+')
    parser.add_argument('--batch_size', '-b', dest='BATCH_SIZE', type=int, default=...)
    parser.add_argument('--lr', help='learning rate', type=float, dest='LR', default=...)
    parser.add_argument('--lr_decay', dest='LR_DECAY', default=False, action='store_true')
    parser.add_argument('--opt', help='optimizer for train, default [adam]',
                        dest='OPT', default='adam', choices=['adam', 'sgd_m', 'sgd'])
    parser.add_argument('--module', default='unetpps')  # todo change name

    module_name = 'train_{}'.format(parse_args('--module', 'unetpps'))
    train_module = importlib.import_module(module_name)
    parser = train_module.args(parser)
    args = parser.parse_args()
    config.apply(args)
    train()
