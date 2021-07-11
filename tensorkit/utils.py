import os
import re
import shutil
import sys
import time


class TimeTic(object):
    def __init__(self) -> None:
        self._pre = {None: time.time()}

    def tic(self, tid=None, unit='s') -> float:
        now = time.time()
        delta = (now - self._pre[tid]) if tid in self._pre.keys() else 0.
        self._pre[tid] = now
        if unit == 's':
            return delta
        elif unit == 'ms':
            return delta * 1e3
        else:
            raise RuntimeError('ERROR: do not support unit: {}'.format(unit))


class BackupFiles(object):
    """"
    default backup all python modules loaded in running time exclude in sys.path and site-packages

    The call of include and exclude for dirs or files should
    be in an order which influences what files will be backuped.

    Example:
        >>> bf = BackupFiles()
        >>> bf.include_dirs('.', suffix='.py', recursive=True)
        >>> bf.exclude_dirs('TEST_DS', 'LOG_DIR', recursive=True)
        >>> bf.run('log/source')
    """

    def __init__(self) -> None:
        super().__init__()
        self._order_collect_dirs = list()
        self._include_files = set()
        self._backup_files = set()

        for m in tuple(sys.modules.values()):
            if hasattr(m, '__file__'):
                p = os.path.abspath(m.__file__)
                fn = file_name(p)
                if fn == '__init__.py':
                    self.include_files(p, target_dir=os.path.join(*m.__name__.split('.')))
                else:
                    target_dir = m.__name__.split('.')[:-1]
                    target_dir = os.path.join(*target_dir) if len(target_dir) > 0 else ''
                    self.include_files(p, target_dir=target_dir)

        [self.exclude_dirs(i, recursive=True) for i in sys.path if os.path.isdir(i) and i != '' and i != '.']

    def include_dirs(self, *include_dirs, target_dir=None, suffix=None, recursive=False):
        """
        backup files with `suffix` in `include_files` to `target_dir`

        :param include_dirs: [include_dir, ...]
        :param target_dir:
        :param suffix: None for all files
        :param recursive:
        :return:
        """
        target_dir = '.' if target_dir is None else target_dir
        self._order_collect_dirs.append(('i', include_dirs, target_dir, suffix, recursive))
        return self

    def exclude_dirs(self, *exclude_dirs, recursive=False):
        """
        not backup files in `exclude_dirs`

        :param exclude_dirs: [exclude_dir, ...]
        :param recursive:
        :return:
        """
        self._order_collect_dirs.append(('e', exclude_dirs, recursive))
        return self

    def include_files(self, *include_files, target_dir=None):
        """
        backup files in `include_files` to `target_dir`

        :param include_files: [include_file, ...]
        :param target_dir:
        :return:
        """
        self._order_collect_dirs.append(('a', include_files, target_dir))
        return self

    def exclude_files(self, *exclude_files):
        """
        not backup files from `exclude_files`

        :param exclude_files: [exclude_file, ...]
        :return:
        """
        self._order_collect_dirs.append(('r', exclude_files))
        return self

    def _handle_include_dirs(self, *include_dirs, target_dir, suffix, recursive):
        """
        :param include_dirs:
        :param target_dir:
        :param suffix:  None for all file
        :param recursive:
        :return:
        """
        for d in include_dirs:
            if not isinstance(d, str) or not os.path.isdir(d):
                continue
            if recursive:
                for rp, _, fns in os.walk(d):
                    fns = [os.path.join(rp, fn) for fn in fns if suffix is None or fn.endswith(suffix)]
                    fns = [os.path.abspath(i) for i in fns]
                    if len(fns) > 0:
                        self._handle_include_files(*fns, target_dir=os.path.join(
                            *normalize_path(os.path.join(target_dir, rp)).split('/')))
            else:
                fns = [i for i in os.listdir(d) if os.path.isfile(i)]
                fns = [os.path.abspath(i) for i in fns]
                self._handle_include_files(*fns, target_dir=os.path.join(target_dir, d))

    def _handle_exclude_dirs(self, *exclude_dirs, recursive):
        new_backup_files = set()

        exclude_dirs = [os.path.abspath(d) for d in exclude_dirs if os.path.isdir(d)]
        exclude_dirs = [d if d.endswith(os.sep) else d + os.sep for d in exclude_dirs]

        for source_f, target_dir in self._backup_files:
            if recursive:
                if all(not source_f.startswith(ed) for ed in exclude_dirs):
                    new_backup_files.add((source_f, target_dir))
            else:
                if parent_path(source_f) not in exclude_dirs:
                    new_backup_files.add((source_f, target_dir))
        self._backup_files = new_backup_files

    def _handle_include_files(self, *include_files, target_dir=None):
        for f in include_files:
            try:
                if os.path.isfile(f):
                    self._backup_files.add((f, target_dir))
            except TypeError:  # os.stat(path) raise
                # TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
                pass

    def _handle_exclude_files(self, *exclude_files):
        new_backup_files = set()
        for source_f, target_dir in self._backup_files:
            if source_f not in exclude_files:
                new_backup_files.add((source_f, target_dir))
        self._backup_files = new_backup_files

    def _collect_backup_files(self):
        for i in self._order_collect_dirs:
            _type, _args = i[0], i[1:]
            if _type == 'a':
                self._handle_include_files(*_args[0], target_dir=_args[-1])
            elif _type == 'r':
                self._handle_exclude_files(*_args[0])
            elif _type == 'i':
                self._handle_include_dirs(*_args[0], target_dir=_args[1], suffix=_args[2], recursive=_args[-1])
            elif _type == 'e':
                self._handle_exclude_dirs(*_args[0], recursive=_args[-1])
            else:
                raise RuntimeError('TODO: type %s' % _type)

    def run(self, root_dir):
        """
        root_dir will be `exclude_dir`
        :param root_dir: root dir for backup
        :return:
        """
        root_dir = os.path.abspath(root_dir)
        if os.path.exists(root_dir):
            assert os.path.isdir(root_dir), 'invalid root_dir: %s' % root_dir
        else:
            mkdir(root_dir, True)

        self._collect_backup_files()

        for source_file, target_dir in self._backup_files:
            target_file = os.path.join(root_dir, target_dir, file_name(source_file))
            if source_file != target_file and not source_file.startswith(root_dir):
                mkdir(parent_path(target_file), True)
                shutil.copyfile(source_file, target_file)


def get_time(format='%Y.%m.%d_%H.%M.%S', seconds=None):
    return time.strftime(format, time.localtime(seconds))


def normalize_path(path):
    p = path.replace('\\', '/').replace('/./', '/')
    p = p[3:] if p.startswith('/../') else p
    p = re.compile(r'/[^./]*?/\.\.').sub('', p)  # remove 'yy/xx/../zz' to 'yy/zz'
    p = re.compile(r'/+').sub('/', p)
    p = p[2:] if p.startswith('./') else p
    p = p.rstrip('/') if len(p) > 1 else p
    return p


def mkdir(dir, recursive=False):
    if os.path.exists(dir):
        return
    pd = parent_path(dir)
    if os.path.exists(pd):
        os.mkdir(dir)
    elif recursive:
        mkdir(pd, recursive)
        os.mkdir(dir)
    else:
        raise RuntimeError('parent dir not exists: %s' % pd)


def file_name(path):
    if os.sep in path:
        return path[path.rindex(os.sep) + 1:]
    return path


def parent_path(path):
    path = os.path.abspath(path)
    return os.path.dirname(path)


if __name__ == '__main__':
    print(normalize_path('/../'))
