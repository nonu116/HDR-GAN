import sys
from typing import Any, Callable

import tensorflow as tf
from tensorflow.python.client import timeline

from tensorkit.log import logger
from tensorkit.restore import Restore
from tensorkit.save import Saver
from tensorkit.sess import session


class StepInfo(object):
    current_iter = -1
    current_epoch_iter = -1
    current_epoch = -1
    epoch_iter = -1
    epoch = -1
    local_iter = -1

    def __str__(self):
        return 'epoch_iter {}/{}, epoch {}/{}, current_iter {}'.format(
            self.current_epoch_iter, self.epoch_iter,
            self.current_epoch, 'Inf' if self.epoch is None else self.epoch,
            self.current_iter
        )

    def is_per(self, epoch_per=None, global_iter_per=None, epoch_iter_per=None):
        ret = self.local_iter != 0
        if epoch_per is not None:
            ret = ret and int(self.current_epoch % epoch_per) == 0
        if global_iter_per is not None:
            ret = ret and int(self.current_iter % global_iter_per) == 0
        if epoch_iter_per is not None:
            ret = ret and int(self.current_epoch_iter % epoch_iter_per) == 0
        return ret


class Trainer(object):
    """
    Example:
        >>> from tensorkit import Config
        >>>
        >>> config = Config()
        >>> model, loss, train_op = ..., ..., ...
        >>> trainer = Trainer() \
        ...     .set_epoch(config.BATCH_SIZE, config.DATASET_SIZE, config.EPOCH) \
        ...     .set_gpu(config.CUDA_VISIBLE_DEVICES, config.ALLOW_GROWTH) \
        ...     .add_train_op([train_op, loss]) \
        ...     .set_summary(config.LOG_DIR, config.SUMMARY_STEP) \
        ...     .set_restore(ckpt_dir=config.LOG_DIR, ckpt_file=config.CKPT_FILE, optimistic=True) \
        ...     .set_saver(model.all_variables(), config.LOG_DIR, 'model', config.SAVE_STEP) \
        ...     .add_prepare_listener(lambda _: model.view_trainable_variables()) \
        ...     .finalize_graph()
        >>> trainer.start()
    """

    def __init__(self):
        self._sess = None
        self._initialize_global_vars = True
        self._step_info = StepInfo()
        self._epoch_iter = 0
        self._epoch = None
        self._prepare_listener = list()
        self._iter_start_listener = list()
        self._on_train_iter_res = list()
        self._finish_listener = list()
        self._train_op = []
        self._feed_dict = dict()
        self._finalize_graph = True
        self._global_step = 0
        self._global_step_var = None
        self._global_step_var_add = None
        self._summary = None
        self._summary_dir = None
        self._summary_merged = None
        self._summary_merge_key = None
        self._summary_writer = None
        self._summary_step = None
        self._val_loss = []
        self._val_feed_dict = dict()
        self._gpu = '-1'
        self._allow_gpu_growth = False
        self._saver = None
        self._saver_step = None
        self._restore = None
        self._timeline = False
        self._timeline_step = None
        self._timeline_fname = None
        self._timeline_run_options = None
        self._timeline_run_metadata = None
        self._to_stop = False

        with tf.variable_scope('tensorkit', reuse=tf.AUTO_REUSE):
            if self._global_step_var is None:
                self._global_step_var = tf.get_variable('tk_global_step', initializer=0., trainable=False)
            if self._global_step_var_add is None:
                self._global_step_var_add = tf.assign_add(self._global_step_var, 1.)

    def set_epoch(self, batch_size: int, dataset_size: int, epoch: float = None):
        assert batch_size > 0 and dataset_size > 0
        assert epoch is None or epoch > 0.
        self._epoch_iter = int(dataset_size / batch_size + .5)
        self._epoch = epoch
        self._step_info.epoch = self._epoch
        self._step_info.epoch_iter = self._epoch_iter
        return self

    def set_gpu(self, gpu, allow_growth=False):
        self._gpu, self._allow_gpu_growth = gpu, allow_growth
        return self

    def add_iter_start_listener(self, func: Callable[[tf.Session, StepInfo], Any]):
        assert callable(func)
        self._iter_start_listener.append(func)
        return self

    def add_prepare_listener(self, func: Callable[[tf.Session], Any]):
        assert callable(func)
        self._prepare_listener.append(func)
        return self

    def add_finish_listener(self, func: Callable[[tf.Session], Any]):
        assert callable(func)
        self._finish_listener.append(func)
        return self

    def on_train_iter_res(self, func: Callable[[tf.Session, Any, Any, StepInfo], Any]):
        """  func: callable = lambda sess, res, val_loss, step_info: None """
        assert callable(func)
        self._on_train_iter_res.append(func)
        return self

    def add_train_op(self, train_op, feed_dict=None):
        self._train_op.append(train_op)
        if feed_dict is not None:
            _keys = self._feed_dict.keys()
            for k in feed_dict.keys():
                if k in _keys:
                    raise RuntimeError('{} should not be feed twice, original {}, new {}'.format(k, self._feed_dict[k],
                                                                                                 feed_dict[k]))
            self._feed_dict.update(feed_dict)
        return self

    def finalize_graph(self, finalize=True):
        self._finalize_graph = finalize
        return self

    def set_summary(self, log_dir, pstep, val_loss=None, feed_dict=None, merge_key=None):
        if pstep > 0:
            self._summary = True
            self._summary_step = pstep
            self._summary_dir = log_dir
            self._summary_merge_key = merge_key if merge_key is not None else tf.GraphKeys.SUMMARIES
            if val_loss is not None:
                self._val_loss.append(val_loss)
            if feed_dict is not None:
                _keys = self._val_feed_dict.keys()
                for k in feed_dict.keys():
                    if k in _keys:
                        raise RuntimeError('{} should not be feed twice, original {}, new {}'.format(
                            k, self._val_feed_dict[k], feed_dict[k]))
                self._val_feed_dict.update(feed_dict)
        else:
            self._summary = False
        return self

    def set_saver(self, var_list, save_dir, save_name, pstep, **kwargs):
        """
        :param var_list: If `None`, defaults to the list of all saveable objects.
        :param save_dir:
        :param save_name:
        :param pstep:
        :param kwargs:
        :return:
        """
        if pstep <= 0:
            return self
        if var_list is not None:
            var_list = self._add_trainer_vars(var_list)
        self._saver = Saver(save_dir=save_dir, model_name=save_name, var_list=var_list, name='tk_saver', **kwargs)
        self._saver_step = pstep
        return self

    def set_timeline(self, pstep, fname='timeline.json'):
        if pstep > 0:
            self._timeline = True
            self._timeline_step = pstep
            self._timeline_fname = fname
            self._timeline_run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._timeline_run_metadata = tf.RunMetadata()
        else:
            self._timeline = False
        return self

    def set_session(self, sess):
        self._sess = sess
        return self

    def initialize_global_variables(self, initialize=True):
        self._initialize_global_vars = initialize
        return self

    def _train_loop(self, sess):
        assert self._train_op is not None, 'train_op can not be None'

        if self._summary:
            self._summary_writer = tf.summary.FileWriter(self._summary_dir, sess.graph)
            assert self._summary_merge_key in sess.graph.get_all_collection_keys(), '{} not in {}'.format(
                self._summary_merge_key, sess.graph.get_all_collection_keys())
            self._summary_merged = tf.summary.merge_all(self._summary_merge_key)

        if self._initialize_global_vars:
            sess.run(tf.global_variables_initializer())

        logger.info('COMMAND: ' + ' '.join(sys.argv))

        if self._restore is not None:
            self._restore.restore(sess)

        for i in self._prepare_listener:
            i(sess)

        self._global_step = int(sess.run(self._global_step_var))
        self._update_step_info()

        if self._finalize_graph:
            sess.graph.finalize()

        _summary_op = (self._train_op, self._val_loss)
        _summary_feed_dict = dict()
        _summary_feed_dict.update(self._feed_dict)
        _summary_feed_dict.update(self._val_feed_dict)
        self._step_info.local_iter = 0
        while self._epoch is None or self._global_step < self._epoch * self._epoch_iter:
            if self._to_stop:
                logger.info('stop iteration at {}'.format(self._step_info))
                break

            for i in self._iter_start_listener:
                i(sess, self._step_info)

            is_timeline_iter, is_summary_iter = self._is_timeline_iter(), self._is_summary_iter()

            if is_timeline_iter and is_summary_iter:
                res, summary_str, self._global_step = sess.run(
                    [_summary_op, self._summary_merged, self._global_step_var_add],
                    feed_dict=_summary_feed_dict,
                    options=self._timeline_run_options, run_metadata=self._timeline_run_metadata)
                tl = timeline.Timeline(self._timeline_run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline_%s.json' % self._global_step, 'w') as f:
                    f.write(ctf)
                self._summary_writer.add_summary(summary_str, self._global_step)
                self._summary_writer.flush()
            elif is_timeline_iter:
                res, self._global_step = sess.run(
                    [self._train_op, self._global_step_var_add],
                    feed_dict=self._feed_dict,
                    options=self._timeline_run_options, run_metadata=self._timeline_run_metadata)
                tl = timeline.Timeline(self._timeline_run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline_%s.json' % self._global_step, 'w') as f:
                    f.write(ctf)
            elif is_summary_iter:
                res, summary_str, self._global_step = sess.run(
                    [_summary_op, self._summary_merged, self._global_step_var_add],
                    feed_dict=_summary_feed_dict)
                self._summary_writer.add_summary(summary_str, self._global_step)
                self._summary_writer.flush()
            else:
                res, self._global_step = sess.run([self._train_op, self._global_step_var_add],
                                                  feed_dict=self._feed_dict)
            self._global_step = int(self._global_step)
            self._update_step_info()

            _res, _val_loss = res if is_summary_iter else (res, None)
            if len(self._train_op) == 1:
                _res = _res[0]
            if is_summary_iter and len(self._val_loss) == 1:
                _val_loss = _val_loss[0]
            for i in self._on_train_iter_res:
                i(sess, _res, _val_loss, self._step_info)

            if self._is_save_iter():
                self._saver.call(sess, self._global_step)

        for i in self._finish_listener:
            i(sess)

        if self._summary:
            self._summary_writer.close()
        print()
        logger.info('finish train, step_info: {}'.format(self._step_info))

    def start(self):
        try:
            if self._sess is None:
                with session(self._gpu, self._allow_gpu_growth) as sess:
                    self._train_loop(sess)
            else:
                self._train_loop(self._sess)
        except(KeyboardInterrupt, SystemExit) as e:
            print()
            logger.error('ERROR: train stop ({}'.format(e))

    def set_restore(self, var_list=None, ckpt_dir=None, ckpt_file=None, optimistic=False):
        """
        :param var_list:    vars for restore. If `None`, defaults to the list of all saveable objects.
        :param ckpt_dir:    prefix of model files.
        :param ckpt_file:   exact name of model file.
        :param optimistic:  only restore weights of same names with model.
        :return:
        """
        if var_list is not None:
            var_list = self._add_trainer_vars(var_list)
        self._restore = Restore().init(var_list, ckpt_dir, ckpt_file, optimistic)
        return self

    def show(self):
        summary = 'SUMMARY_DIR: %s' % self._summary_dir
        summary += '\nSUMMARY_STEP: %s' % self._summary_step
        model_save = str(self._saver)
        model_save += '\nSAVE_STEP: %s' % self._saver_step
        restore = str(self._restore)
        epoch = 'EPOCH: %s' % self._epoch if self._epoch_iter is not None else 'EPOCH: inf'
        epoch += '\nEPOCH_ITER: %s' % self._epoch_iter
        finalize_graph = 'FINALIZE_GRAPH: %s' % self._finalize_graph

        content = '\n'.join([summary, model_save, restore, epoch, finalize_graph])
        content = '\n'.join([' trainer.configure '.center(70, '='), content, ''.center(70, '=')])
        logger.info(content)

    def to_stop(self, stop=True):
        self._to_stop = stop
        return self

    def _add_trainer_vars(self, var_list):
        def add_var(_var, _list):
            key = _var.name
            if isinstance(_list, dict):
                if key in _list.keys() and _list[key] != _var:
                    raise RuntimeError('key is registered by {}'.format(_list[key]))
                _list[key] = _var
            elif isinstance(_list, list):
                if _var not in _list:
                    _list.append(_var)
            else:
                raise RuntimeError('invalid type for var_list: {}'.format(type(var_list)))
            return _list

        return add_var(self._global_step_var, var_list)

    def _update_step_info(self):
        self._step_info.current_iter = int(self._global_step)
        self._step_info.current_epoch = int(self._global_step / self._epoch_iter)
        self._step_info.current_epoch_iter = int(self._global_step % self._epoch_iter)
        self._step_info.local_iter += 1

    def _is_timeline_iter(self):
        return self._timeline and \
               self._step_info.local_iter != 0 and \
               self._step_info.is_per(global_iter_per=self._timeline_step)

    def _is_summary_iter(self):
        return self._summary and (self._step_info.local_iter == 0 or
                                  self._step_info.is_per(global_iter_per=self._summary_step))

    def _is_save_iter(self):
        return self._saver is not None and \
               self._step_info.local_iter != 0 and \
               self._step_info.is_per(global_iter_per=self._saver_step)
