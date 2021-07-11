import os
import tensorflow as tf


class AbstractSaver(object):

    def __init__(self, save_dir, model_name, var_list=None, name=None, **kwargs) -> None:
        self._save_dir = save_dir
        self._save_model_name = model_name
        self._saver = tf.train.Saver(var_list=var_list, name=name, **kwargs)

    def call(self, *args, **kwargs) -> None:
        raise RuntimeError("Don't use this abstract class")

    def _save(self, sess, global_step) -> None:
        self._saver.save(sess, os.path.join(self._save_dir, self._save_model_name), global_step=global_step)

    def __str__(self):
        return 'SAVE_FILE: {}'.format(os.path.join(self._save_dir, self._save_model_name))


class Saver(AbstractSaver):

    def call(self, sess, global_step) -> None:
        self._save(sess, global_step)


class OptimalSaver(AbstractSaver):

    def __init__(self, loss_history_count, save_dir, model_name, var_list=None, min_freq=100, name=None, **kwargs) \
            -> None:
        """
        Saving variables when history_loss has minimum sum.
        :param loss_history_count: How many continue history loss to be added up.
        :param save_dir:
        :param model_name:
        :param var_list:
        :param min_freq: Minimum saving frequency to ovoid saving variables frequently.
        :param name:
        """
        super().__init__(save_dir, model_name, var_list, name, **kwargs)
        if var_list is not None:
            assert len(var_list) > 0, 'invalid var_list: {}'.format(var_list)
        assert min_freq > 0, 'invalid min_freq: {}'.format(min_freq)
        self._history_count = loss_history_count
        self._loss_history = [1e6 * loss_history_count] * loss_history_count
        self._loss_history_sum = sum(self._loss_history)
        self._loss_history_min = self._loss_history_sum
        self._min_freq = min_freq
        self._cooling_period = loss_history_count

    def call(self, sess, loss, global_step) -> None:
        """
        update loss_history and save model if arrive minial loss_history
        :param sess:
        :param loss: loss for record, should be scale not Tensor
        :param global_step:
        :return:
        """
        self._loss_history_sum = self._loss_history_sum - self._loss_history[
            global_step % self._history_count] + loss
        self._loss_history[global_step % self._history_count] = loss
        self._cooling_period -= 1

        if self._loss_history_sum < self._loss_history_min and self._cooling_period <= 0:
            self._save(sess, global_step)
            self._loss_history_min = self._loss_history_sum
            self._cooling_period = self._min_freq
