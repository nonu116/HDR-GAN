import os

import tensorflow as tf

from tensorkit.log import logger, Color


class Restore(object):
    def __init__(self):
        self._var_list = None
        self._restore_saver = None
        self._restore_optimistic = False
        self.restore_ckpt_file = None
        self._inited = False

    def init(self, var_list=None, ckpt_dir=None, ckpt_file=None, optimistic=False):
        """
        :param var_list:    vars for restore
        :param ckpt_dir:    prefix of model files.
        :param ckpt_file:   exact name of model file, priority is higher than `ckpt_dir`
        :param optimistic:  only restore weights of same names with model.
        :return:
        """
        assert (var_list is None) or (len(var_list) > 0), 'invalid var_list: {}'.format(var_list)
        assert ckpt_dir is not None or ckpt_file is not None, 'ckpt_dir and ckpt_file are both None'
        self._var_list = var_list
        self._restore_optimistic = optimistic
        if ckpt_file is None:
            assert os.path.exists(ckpt_dir), 'invalid checkpoint dir: %s' % ckpt_dir
            # get ckpt file.
            self.restore_ckpt_file = tf.train.latest_checkpoint(os.path.dirname(ckpt_dir + os.sep))
        else:
            self.restore_ckpt_file = ckpt_file
        self._inited = True
        return self

    def restore(self, sess):
        assert self._inited, 'make sure init() before restore()'
        if self._restore_vars(sess):
            logger.info('- succeed restore variables from: {}'.format(self.restore_ckpt_file))
            return True
        return False

    def _restore_vars(self, sess):
        """
        :param sess:
        :return: boolean for successful or not
        """
        if not self._restore_optimistic:
            if self.restore_ckpt_file is None:
                logger.warn(
                    Color.yellow('No checkpoint file for restore vars, checkpoint file is None', bold=True))
                return False
            self._restore_saver = tf.train.Saver(self._var_list, name='tk_restore')
            self._restore_saver.restore(sess, self.restore_ckpt_file)
            return True
        else:
            return self._optimistic_restore_model(sess)

    def _optimistic_restore_model(self, sess):
        """
        restore weights of same names with model.
        :param sess:
        :return:
        """
        if self.restore_ckpt_file is None:
            logger.warn(Color.yellow('No ckpt file for restore vars, ckpt file is None'))
            return False
        reader = tf.train.NewCheckpointReader(self.restore_ckpt_file)
        saved_shapes = reader.get_variable_to_shape_map()
        if self._var_list is None:
            restore_key2vars = {var.name.split(':')[0]: var for var in tf.global_variables()}
        elif isinstance(self._var_list, list):
            restore_key2vars = {var.name.split(':')[0]: var for var in self._var_list}
        elif isinstance(self._var_list, dict):
            restore_key2vars = self._var_list
        else:
            raise RuntimeError('type error {}'.format(self._var_list))
        assert len(restore_key2vars) > 0
        restore_key2vars = sorted([(k, v) for k, v in restore_key2vars.items() if k in saved_shapes])
        msg = []
        var_list = dict()
        with tf.variable_scope('', reuse=True):
            for key, var in restore_key2vars:
                var_shape = var.get_shape().as_list()
                if var_shape == saved_shapes[key]:
                    var_list[key] = var
                    var_name = var.name[:var.name.index(':')]
                    msg.append('- restoring variable: {}'.format(var_name)
                               if var_name == key else
                               '- restoring variable {} from {}'.format(var_name, key))
                else:
                    msg.append(Color.yellow(
                        '- variable({}) with inconsistent shape: {}(graph) != {}(ckpt)'.format(
                            key, var_shape, saved_shapes[key])
                    ))
        if len(var_list) != 0:
            msg += ['- total variable count: {}'.format(len(var_list))]
            logger.info('\n'.join(msg))
            saver = tf.train.Saver(var_list, name='tk_restore')
            saver.restore(sess, self.restore_ckpt_file)
            return True
        else:
            logger.warn(Color.yellow('No vars need to restore from file: {}'.format(self.restore_ckpt_file)))
        return False

    def __str__(self):
        content = 'RESTORE_OPTIMISTIC: %s' \
                  '\nRESTORE_CHECKPOINT_FILE: %s' % (self._restore_optimistic, self.restore_ckpt_file)
        return content
