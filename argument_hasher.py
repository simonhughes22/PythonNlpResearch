__author__ = 'simon.hughes'

from collections import defaultdict

def argument_hasher(value):
    if type(value) == list or type(value) == tuple:
        return "_".join(map(argument_hasher, value))
    elif type(value) == dict or type(value) == defaultdict:
        return "_".join(map(argument_hasher, value.items()))
    elif hasattr(value, '__call__'):
        return value.func_name
    return str(value)