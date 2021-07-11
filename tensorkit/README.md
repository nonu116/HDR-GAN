## Tensorkit 

Kit for tensorflow


### train example
```python
import tensorflow as tf

from tensorkit import TimeTic
from tensorkit import Trainer


def graph():
    train_vars, loss, net = ..., ..., ...
    return train_vars, loss, net


def train(config):
    train_vars, loss, net = graph()

    optimizer = tf.train.AdamOptimizer(config.LR, beta1=config.BETA1)
    train_op = optimizer.minimize(loss, var_list=train_vars)

    trainer = Trainer() \
        .set_epoch(config.BATCH_SIZE, config.DATASET_SIZE, config.EPOCH) \
        .set_gpu(config.CUDA_VISIBLE_DEVICES, config.ALLOW_GROWTH) \
        .add_train_op([train_op, loss]) \
        .set_summary(config.LOG_DIR, config.SUMMARY_STEP) \
        .set_restore(ckpt_dir=config.LOG_DIR, ckpt_file=config.CKPT_FILE, optimistic=True) \
        .set_saver(tf.global_variables(), config.LOG_DIR, 'model', config.SAVE_STEP) \
        .add_prepare_listener(lambda _: net.view_trainable_variables()) \
        .finalize_graph()

    tic = TimeTic()

    def progress(sess, res, step_info):
        _, loss_np = res
        print("\r Epoch: %2d/%d,  iter: %4d/%d, time: %4.4f, g_loss: %.8f" % (
            step_info.current_epoch, step_info.epoch,
            step_info.current_epoch_iter, step_info.epoch_iter, tic.tic(), loss_np), end='')

    trainer.on_train_iter_res(progress)
    trainer.start()
```