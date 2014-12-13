"""
    from http://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods
"""

import time
import os
import cPickle as pickle
from collections import defaultdict
import hashlib

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        MAX_ARG_STR = 100

        def append_args(a, s):
            str_arg = str(a)
            if len(str_arg) <= MAX_ARG_STR:
                return s + str_arg + ", "
            else:
                return s + "..., "

        args_str = ""
        for a in args:
            args_str = append_args(a, args_str)
        for k,v in kw.items():
            args_str = append_args(str(k) + " : " + str(v), args_str)

        if len(args_str) > MAX_ARG_STR:
            args_str = " "
        else:
            args_str = "(" + args_str[:-2] + ") "

        # Don't print huge args list
        print 'timeit: %s%stook %2.2f secs' % \
            (method.__name__, args_str, te - ts)
        return result

    return timed


class __memodict_(dict):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        ret = self[key] = self.f(*key)
        return ret

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    return __memodict_(f)

class memoize_to_disk(object):

    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix

    def __call__(self, f):
        # decorate f
        def wrapped_f(**kwargs):
            pickle_key = "_".join(map(lambda (k, v): k + "_" + self.__value2str__(v), sorted(kwargs.items())))
            pickle_file = self.filename_prefix + pickle_key
            if os.path.exists(pickle_file):
                return pickle.load(open(pickle_file, "r+"))
            result = f(**kwargs)
            pickle.dump(result, open(pickle_file, "w+"))
            return result
        return wrapped_f

    def __value2str__(self, value):
        if type(value) == list or type(value) == tuple \
                or type(value) == dict or type(value) == defaultdict:
            return "_".join(map(self.__value2str__, value))
        elif hasattr(value, '__call__'):
            return value.func_name
        return str(value)

if __name__ == "__main__":

    import time

    """
    @timeit
    def a_function(a,b,c):
        time.sleep(1)

    a_function(range(10), "asass", 210921.982198)
    """

    @memoize
    def foo(a, b):
        return a * b

    @memoize
    def bar(a,b):
        return a + b

    print "foo"
    print foo(4, 2)
    print foo(4, 2)
    print foo

    print "bar"
    print bar(4, 2)
    print bar(4, 2)
    print bar

    print "foo", foo, "bar", bar

    print "foo"
    print foo('xo', 3)
    print foo('xo', 3)
    print foo

    print "bar"
    print bar('xo', "3")
    print bar('xo', "3")
    print bar
