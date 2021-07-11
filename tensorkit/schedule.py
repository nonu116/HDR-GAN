from tensorkit.train import StepInfo

__schedule = dict()
__schedule_order = dict()

__UNITS = ['iter', 'epoch']
__ORDER_COUNT = 0


def register(func_or_op, start, per, end=None, feed_dict=None, unit='iter', order=False):
    unit = unit.lower()
    assert unit in __UNITS
    key = (start, end, per, unit)
    if order:
        global __ORDER_COUNT
        __schedule_order.setdefault(key, [])
        __schedule_order[key].append((func_or_op, feed_dict, __ORDER_COUNT))
        __ORDER_COUNT += 1
    else:
        __schedule.setdefault(key, [])
        value = (func_or_op, feed_dict)
        if feed_dict is not None:
            listeners = __schedule[key]
            fd_keys = set()
            for _op, _fd in listeners:
                if _fd is not None:
                    for fdk in _fd.keys():
                        fd_keys.add(fdk)
            assert all(i not in fd_keys for i in feed_dict.keys()), 'duplicate feed_dict: {}\n{}'.format(feed_dict.keys,
                                                                                                         fd_keys)
        __schedule[key].append(value)


def __handle_schedule(sess, step_info: StepInfo, order):
    _sch = __schedule_order if order else __schedule
    keys = list(_sch.keys())
    listeners = []
    for k in keys:
        start, end, per, unit = k
        current = step_info.current_iter if unit == 'iter' else step_info.current_epoch
        if current < start:
            continue
        if end is not None and end < current:
            continue
        if int(current % per) == 0:
            listeners.extend(_sch[k])

    if order:
        listeners = sorted(listeners, key=lambda i: i[-1])
        for func_op, fd, _ in listeners:
            if callable(func_op):
                func_op(sess, step_info)
            else:
                sess.run(func_op) if fd is None else sess.run(func_op, fd)
    else:
        ops = []
        fds = dict()
        for func_op, fd in listeners:
            if callable(func_op):
                func_op(sess, step_info)
            else:
                ops.append(func_op)
                if fd is not None:
                    assert all(k not in fds.keys() for k in fd.keys()), 'duplicate feed_dict: {}\n{}'.format(fd.keys(),
                                                                                                             fds.keys())
                    fds.update(fd)
        if len(ops) > 0:
            sess.run(ops) if len(fds) == 0 else sess.run(ops, feed_dict=fds)


def handle_schedule(sess, step_info: StepInfo):
    __handle_schedule(sess, step_info, False)
    __handle_schedule(sess, step_info, True)
