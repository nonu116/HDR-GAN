import numpy as np
import tensorflow as tf

from tensorkit.log import logger, Color


class BaseModel(object):
    def __init__(self, name=None):
        self.name = name if name is not None else self.__class__.__name__

    def graph(self, *args, **kwargs):
        raise NotImplementedError

    def all_variables(self):
        return tf.global_variables(self.name)

    def trainable_variables(self):
        return tf.trainable_variables(self.name)

    def view_trainable_variables(self):
        content = (' %s ' % self.name).center(100, '=')
        variables = []
        var_str = []
        p_nums = []
        for i in self.trainable_variables():
            p_num = np.prod(i.get_shape().as_list())
            p_nums.append(p_num)
            variables.append(i)
        if len(p_nums) == 0:
            logger.info(content)
            logger.info('None\n' + ''.center(100, '='))
            return
        total_nums = np.sum(p_nums)
        pers = [100. * p / total_nums for p in p_nums]
        pers_mean = np.mean(pers)
        pers_fmt = '{:>5.2f}%'
        fmt = r'({}, {:' + str(len(str(np.max(p_nums)))) + 'd}) {}: {}'
        for per, p, v in zip(pers, p_nums, variables):
            pers_str = pers_fmt.format(per)
            if per > pers_mean:
                pers_str = Color.cyan(pers_str)
            var_str.append(fmt.format(pers_str, p, v.name, v.shape))
        var_str = '\n'.join(var_str)
        content += '\n' + var_str
        content += '\nTotal num of weight: {}, variables count: {}'.format(total_nums, len(variables))
        content += '\n' + ''.center(100, '=')
        logger.info(content)
