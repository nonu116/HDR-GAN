import os

import yaml
from yaml.representer import SafeRepresenter


class DictWrapper(dict):
    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        return self.__delitem__(item)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, dict) and value.__class__ is not DictWrapper:
            value = DictWrapper(value)
            self[item] = value
        return value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and value.__class__ is not DictWrapper:
            value = DictWrapper(value)
        super().__setitem__(key, value)


yaml.add_representer(DictWrapper, SafeRepresenter.represent_dict, yaml.SafeDumper)


class Config(object):
    def __init__(self, fname=None, args=None):
        self._dict = DictWrapper()
        self.__setattr_ori = self.__setattr_with_check

        if fname is not None:
            self.load(fname)

        if args is not None:
            self.apply(args)

    def dump(self, file_name, mode='w'):
        """
        Serialize Config into YAML file
        :param file_name: YAML file
        :param mode:
        :return:
        """
        with open(file_name, mode) as f:
            yaml.safe_dump(self._dict, f, default_flow_style=False)

    def load(self, file_name):
        """
        Parse Config from YAML document
        :param file_name:
        :return:
        """
        assert os.path.exists(file_name), 'ERROR: config file not found (%s)' % file_name
        with open(file_name, 'r') as f:
            self.apply(yaml.safe_load(f))

    def apply(self, args):
        """
        parse from object's attr or dict's key, dot in attr will be split as a level
        :param args: args from argparse or a object or a dict.
        :return:
        """
        if hasattr(args, '__dict__') and isinstance(args.__dict__, dict):
            dic = args.__dict__
        elif isinstance(args, dict):
            dic = args
        else:
            raise RuntimeError('args object should have  __dict__ attr or a dict is needed: {}'.format(args))

        for k, v in dic.items():
            if v is not ...:
                ks = k.split('.')
                self.__raise_if_key_in_dir(ks[0])
                if len(ks) == 1:
                    self._dict[ks[-1]] = v
                else:
                    obj = self._dict
                    for ki in ks[:-1]:
                        if ki not in obj.keys():
                            obj[ki] = DictWrapper()
                        obj = obj[ki]
                    obj[ks[-1]] = v

    def __getattr__(self, item):
        return self._dict[item]

    def __setattr__(self, key, value):
        self.__setattr_ori(key, value)

    def __delattr__(self, item):
        self.__raise_if_key_in_dir(item)
        return self._dict.__delattr__(item)

    def __setattr_ori(self, key, value):
        super().__setattr__(key, value)

    def __raise_if_key_in_dir(self, key):
        attr_or_func = dir(self)
        if key in attr_or_func:
            raise RuntimeError('key should not be in names of object\'s attr or function. invalid key: {}'.format(key))

    def __setattr_with_check(self, key, value):
        self.__raise_if_key_in_dir(key)
        self._dict[key] = value

    def __str__(self) -> str:
        content = ' configure '.center(60, '=') + '\n'
        keys = sorted(list(self._dict.keys()))
        for k in keys:
            content += '{}: {}\n'.format(left(k, 24), self._dict[k])
        content += ''.center(60, '=')
        return content

    def __eq__(self, other):
        if type(self) == type(other):
            return self._dict.__eq__(other._dict)
        return False

    def safely_get(self, attr, default=None):
        try:
            return self.__getattr__(attr)
        except KeyError:
            return default

    def show(self):
        print(self)


def left(string, length):
    lg = len(string)
    return string if lg >= length else string + ' ' * (length - lg)
