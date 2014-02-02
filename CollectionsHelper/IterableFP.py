__author__ = 'simon.hughes'

""" Private """
def __get_fn__(collection):
    return type(collection)

""" Public """
def head(collection, n=1):
    """ Return the first item, or first n items
    """
    if len(collection) == 0:
        return None
    lst_col = list(collection)
    if n == 1:
        return lst_col[0]
    return lst_col[0:n]

def sample(collection):
    """ Sample the first 10 elements of the collection
    """
    return head(collection, 10)

def tail(collection, n=1):
    if len(collection) == 0:
        return None

    lst_col = list(collection)
    if n == 1:
        return lst_col[-1]
    return lst_col[-(n):]

def compact(collection):
    fn = __get_fn__(collection)
    return fn((i for i in collection if bool(i)))

def count(predicate, collection):
    """ How many elements match the predicate
    """
    return len(filter(predicate, collection))

def index_of(predicate, collection):
    """ Finds the index of the first matching items in a collection
    """
    for i, item in enumerate(collection):
        if predicate(item):
            return i
    return -1
