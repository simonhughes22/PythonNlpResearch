__author__ = 'simon.hughes'

from ReflectionUtils import is_iterable, is_callable
import inspect

""" Private """
def __get_fn__(iterable):
    return type(iterable)

""" Public """
def head(iterable, n=1):
    """ Return the first item, or first n items
    """
    lst_col = list(iterable)
    if len(lst_col) == 0:
        return None
    if n == 1:
        return lst_col[0]
    return lst_col[0:n]

def sample(iterable):
    """ Sample the first 10 elements of the iterable
    """
    return head(iterable, 10)

def tail(iterable, n=1):

    lst_col = list(iterable)
    if len(lst_col) == 0:
        return None

    if n == 1:
        return lst_col[-1]
    return lst_col[-(n):]

def compact(iterable):
    """ Removes falsey items
    """
    fn = __get_fn__(iterable)
    return fn((i for i in iterable if bool(i)))

def count(predicate, iterable):
    """ How many elements match the predicate
    """
    return len(filter(predicate, iterable))

def index_of(predicate, iterable):
    """ Finds the index of the first matching items in a iterable
    """
    for i, item in enumerate(iterable):
        if predicate(item):
            return i
    return -1

def merge(iterables):
    """ Merges a number of iterables into one unique,
        sorted list
    """
    if len(iterables) == 0:
        return []

    merged = set()
    for iterable in iterables:
        merged.update(iterable)

    return sorted(merged)

def join(iterable_a, iterable_b, where = lambda a,b: a == b, select = lambda a,b: (a,b)):
    """ Joins items from two iterables where they match in the where fn
        select() determines the value returned
    """
    fn = __get_fn__(iterable_a)
    matches = []
    for a in iterable_a:
        for b in iterable_b:
            if where(a,b):
                matches.append(select(a,b))

    return fn(matches)

def recursive_map(fn, iterable):
    """
    fn          :   fn (callable) that takes a non-iterable
    iterable    :   some iterable of (possible) iterables

    returns     :   list of list of ... after applying fn to each nested iterable

    Applies fn to each iterable within iterable, recursing through all nested iterables
    """
    if not is_callable(fn):
        raise Exception("Parameter fn is NOT callable")

    if not is_iterable(iterable):
        raise Exception("Parameter iterable is NOT iterable")

    """ The true power of recursion!
    """
    def rec_map(potential_iterable):
        if is_iterable(potential_iterable):
            return map(rec_map, potential_iterable)
        return fn(potential_iterable)

    return map(rec_map, iterable)

def recursive_reduce(fn, iterable):
    """
    fn          :   fn (callable) that takes an iterable
    iterable    :   some iterable of (possible) iterables

    returns     :   list of list of ... after applying fn to each nested iterable

    Applies fn to each iterable within iterable, reducing nested iterables
    """
    if not is_callable(fn):
        raise Exception("Parameter fn is NOT callable")

    if not is_iterable(iterable):
        raise Exception("Parameter iterable is NOT iterable")

    """ The true power of recursion!
    """
    def rec_red(potential_iterable):
        lst = []
        for i in potential_iterable:
            if is_iterable(i):
                lst.append(rec_red(i))
            else:
                lst.append(i)
        return fn(lst)
    return rec_red(iterable)

def flatten(iterable):
    """
    iterable    :   some iterable of (possible) iterables

    returns     :   list of list of ... after applying fn to each nested iterable

    Applies fn to each iterable within iterable, recursing through all nested iterables
    """
    if not is_iterable(iterable):
        raise Exception("Parameter iterable is NOT iterable")

    """ The true power of recursion!
    """
    def fltn(potential_iterable):
        lst = []
        for i in potential_iterable:
            if is_iterable(i):
                lst.extend(fltn(i))
            else:
                lst.append(i)
        return lst
    return fltn(iterable)

def recursive_print(nested_iterable, padding_str="\t"):
    """
    iterable    :   some iterable of (possible) iterables
    padding_str :   str
                        used to indent nested iterables

    Prints each nested iterable, with padding_str * level of nesting
    """
    def r_print(level, iterble):
        if is_iterable(iterble):
            for i in iterble:
                r_print(level + 1, i)
        else:
            print(padding_str * level, str(iterble))
    r_print(-1, nested_iterable)

""" Aliases """
first = head
last = tail

if __name__ == "__main__":

    l = [
        9,
        [2, 3, 4, 5],
        [
            [6, 7, 8, 9, 10],
            2, 3, 4, [2, 3, 4, 999]],
        1
    ]

    print("Before Mapping")
    recursive_print(l)

    print("\nAfter Mapping")
    str_l = recursive_map(str, l)
    mapped = recursive_map(lambda i: i * 2, str_l)
    recursive_print(mapped)

    print("\nReduced (max)")
    print(str(recursive_reduce(max, l)))
    print("\nReduced (min)")
    print(str(recursive_reduce(min, l)))
