import logging
import os
import sys

from tensorkit.utils import mkdir, parent_path


class Color(object):
    _gray = 30
    _red = 31
    _green = 32
    _yellow = 33
    _blue = 34
    _magenta = 35
    _cyan = 36
    _white = 37
    _crimson = 38

    @staticmethod
    def gray(string, bold=False, highlight=False):
        return Color._colorize(string, Color._gray, bold=bold, highlight=highlight)

    @staticmethod
    def red(string, bold=False, highlight=False):
        return Color._colorize(string, Color._red, bold=bold, highlight=highlight)

    @staticmethod
    def green(string, bold=False, highlight=False):
        return Color._colorize(string, Color._green, bold=bold, highlight=highlight)

    @staticmethod
    def yellow(string, bold=False, highlight=False):
        return Color._colorize(string, Color._yellow, bold=bold, highlight=highlight)

    @staticmethod
    def blue(string, bold=False, highlight=False):
        return Color._colorize(string, Color._blue, bold=bold, highlight=highlight)

    @staticmethod
    def magenta(string, bold=False, highlight=False):
        return Color._colorize(string, Color._magenta, bold=bold, highlight=highlight)

    @staticmethod
    def cyan(string, bold=False, highlight=False):
        return Color._colorize(string, Color._cyan, bold=bold, highlight=highlight)

    @staticmethod
    def white(string, bold=False, highlight=False):
        return Color._colorize(string, Color._white, bold=bold, highlight=highlight)

    @staticmethod
    def crimson(string, bold=False, highlight=False):
        return Color._colorize(string, Color._crimson, bold=bold, highlight=highlight)

    @staticmethod
    def _colorize(string, color_code: int, bold, highlight):
        attr = []
        if highlight:
            color_code += 10
        attr.append(str(color_code))
        if bold:
            attr.append('1')
        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class MulLinesFormatter(logging.Formatter):
    def __init__(self, fmt=None, fmt_mid=None, fmt_last=None, datefmt=None):
        super(MulLinesFormatter, self).__init__(fmt, datefmt=datefmt)
        self._fmt_mid = fmt_mid if fmt_mid is not None else fmt
        self._fmt_last = fmt_last if fmt_last is not None else fmt

    def formatMessage(self, record):
        record_dics = []
        message = record.message
        record_dic = record.__dict__
        for i in message.split('\n'):
            tmp = dict(record_dic)
            tmp['message'] = i
            record_dics.append(tmp)

        res = []
        lines = len(record_dics)
        for ind, r in enumerate(record_dics):
            if ind == 0:
                res.append(self._fmt % r)
            elif ind == lines - 1:
                res.append(self._fmt_last % r)
            else:
                res.append(self._fmt_mid % r)

        return '\n'.join(res)


logger = logging.getLogger('tensorkit')
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_formatter = MulLinesFormatter(fmt=Color.green('[P%(process)d|T%(asctime)s @%(filename)s]') + ' %(message)s',
                               fmt_mid=Color.green('[P%(process)d|T%(asctime)s @%(filename)s') +
                                       Color.green(']', bold=True) + ' %(message)s',
                               fmt_last=Color.green('[P%(process)d|T%(asctime)s @%(filename)s') +
                                        Color.green(']', bold=True) + ' %(message)s',
                               datefmt='%Y.%m.%d_%H:%M:%S')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)


def logging_to_file(filename, with_color=True):
    if not os.path.exists(parent_path(filename)):
        mkdir(parent_path(filename), True)

    handler = logging.FileHandler(filename=filename)
    handler.setLevel(logging.DEBUG)
    if with_color:
        handler.setFormatter(_formatter)
    else:
        handler.setFormatter(MulLinesFormatter(fmt='[P%(process)d|T%(asctime)s @%(filename)s] %(message)s',
                                               fmt_mid='[P%(process)d|T%(asctime)s @%(filename)s] %(message)s',
                                               fmt_last='[P%(process)d|T%(asctime)s @%(filename)s] %(message)s',
                                               datefmt='%Y.%m.%d_%H:%M:%S'))
    logger.addHandler(handler)
