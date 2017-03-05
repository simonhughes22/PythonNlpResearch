"""
    from http://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods
"""

import time
import os
try:
    import cPickle as pickle
except:
    import pickle

from collections import defaultdict
from argument_hasher import argument_hasher
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
        print('timeit: %s%stook %2.2f secs' % \
            (method.__name__, args_str, te - ts))
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

class ArgHashMixin(object):
    def hash_args(self, *args, **kwargs):
        hargs = "_".join(map(lambda v: "_" + self.__value2str__(v), args))
        hkwargs = "_".join(map(lambda tpl: tpl[0] + "_" + self.__value2str__(tpl[1]), sorted(kwargs.items())))
        return hargs + "|" + hkwargs

    def __value2str__(self, value):
        if type(value) == list or type(value) == tuple \
                or type(value) == dict or type(value) == defaultdict:
            return "_".join(map(self.__value2str__, value))
        elif hasattr(value, '__call__'):
            return value.func_name
        return str(value)

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memo_dict(dict, ArgHashMixin):

        def __init__(self, func):
            self.func = func
            self.method_type = "funcation"
            self.obj, self.cls, = None, None

        def __get__(self, obj=None, cls=None):
            # It is executed when decorated func is referenced as a method: cls.func or obj.func.

            if self.obj == obj and self.cls == cls:
                return self  # Use the same instance that is already processed by previous call to this __get__().

            method_type = (
                'staticmethod' if isinstance(self.func, staticmethod) else
                'classmethod' if isinstance(self.func, classmethod) else
                'instancemethod'
                # No branch for plain function - correct method_type for it is already set in __init__() defaults.
            )
            self.obj = obj
            self.cls = cls
            self.method_type = method_type
            return self

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return self[self.hash_args(*args, **kwargs)]

        def __missing__(self, key):
            if self.method_type == "instancemethod":
                self.args = [self.obj] + list(self.args)
                ret = self[key] = self.func(*self.args, **self.kwargs)
            else:
                ret = self[key] = self.func(*self.args, **self.kwargs)
            return ret
    return memo_dict(f)

class memoize_to_disk(object):

    def __init__(self, filename_prefix, verbose=True):
        self.filename_prefix = filename_prefix
        self.verbose = verbose

    def __call__(self, f):
        # decorate f
        def wrapped_f(*args, **kwargs):
            # don't hash args, just kwargs
            s_pickle_key = "_".join(map(lambda tpl: tpl[0] + "_" + self.__value2str__(tpl[1]), sorted(kwargs.items())))
            # hash long filesnames
            #if len(pickle_key) > 225:
            if self.verbose:
                print("Pickle Key:", s_pickle_key)

            pickle_key = str(hash(s_pickle_key))
            pickle_file = self.filename_prefix + pickle_key
            if os.path.exists(pickle_file):
                return pickle.load(open(pickle_file, "r+"))
            result = f(*args, **kwargs)
            pickle.dump(result, open(pickle_file, "w+"))
            return result
        return wrapped_f

    def __value2str__(self, value):
        return argument_hasher(value)

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

    print("foo")
    print(foo(4, 2))
    print(foo(4, 2))
    print(foo)

    print("bar")
    print(bar(4, 2))
    print(bar(4, 2))
    print(bar)

    print("foo", foo, "bar", bar)

    print("foo")
    print(foo('xo', 3))
    print(foo('xo', 3))
    print(foo)

    print("bar")
    print(bar('xo', "3"))
    print(bar('xo', "3"))
    print(bar)
