import inspect

def get_props(o):
    return set(dir(o))

def is_iterable(o):
    return type(o) != str and "__iter__" in get_props(o)

def is_callable(o):
    return inspect.isfunction(o) or inspect.isclass(o) or "__call__" in get_props(o)

def has_props(*keys):
    def closure_single_prop(o):
        props = get_props(o)
        return keys in props

    if len(keys) == 1:
        return closure_single_prop

    def closure_multiple_props(o):
        props = get_props(o)
        for k in keys:
            if k not in props:
                return False
        return True
    return closure_multiple_props
