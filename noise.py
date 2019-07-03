import numbers
import threading
from queue import Queue
import numpy as np
import math
import ctypes
import multiprocessing


class SharedNoiseTable(object):
    def __init__(self):
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        print('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        print('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


class ConstantSchedule(object):
    def __init__(self, value):
        self._value = value

    def value(self, **kwargs):
        return self._value


class LinearSchedule(object):
    def __init__(self, schedule, final_p, initial_p, field):
        self.schedule = schedule
        self.field = field
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, **kwargs):
        assert self.field in kwargs, "Argument {} not provided to scheduler Available: {}".format(self.field, kwargs)
        fraction = min(float(kwargs[self.field]) / self.schedule, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExponentialSchedule(object):
    def __init__(self, initial_p, final_p, schedule, field):
        self.initial_p = initial_p
        self.final_p = final_p
        self.schedule = schedule
        self.field = field

        self.linear = LinearSchedule(
                initial_p=math.log(self.initial_p),
                final_p=math.log(self.final_p),
                schedule=self.schedule,
                field=self.field)

    def value(self, **kwargs):
        return math.exp(self.linear(**kwargs))


def make_schedule(args):
    if isinstance(args, numbers.Number):
        return ConstantSchedule(args)
    else:
        return globals()[args['type']](**{key: value for key, value in args.items() if key != 'type'})