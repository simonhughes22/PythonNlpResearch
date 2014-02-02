from IterableFP import recursive_map

class IdGenerator(object):
    '''
    Generates unique id's for a given hashable items.
    If an id has already been requested for an item, it
    is returned instead
    '''

    def __init__(self, seed=0):
        """
            seed :  initial id value
        """
        
        self.__val2id__ = dict()
        self.__id2val__ = dict()
        self.__counter__ = seed - 1 # init to -1 as we increment before assigning

    def get_id(self, val):
        """ Given a value, generate (or re-use) a unique identifier """
        if val in self.__val2id__:
            return self.__val2id__[val]
        self.__counter__ += 1
        self.__val2id__[val] = self.__counter__
        self.__id2val__[self.__counter__] = val
        return self.__counter__
        
    def get_id2val(self):
        """ Return a copy of the dictionary to prevent updating state"""
        return dict(self.__id2val__.values())

    def get_ids(self):
        """ Get all current ids """
        return self.__id2val__.keys()

    def get_key(self, id):
        """ Given an id, recover the original value """
        return self.__id2val__[id]

    def max_id(self):
        return max(self.__id2val__.keys())

    def iterable2ids(self, iterable):
        """
        iterable    :   iterable (of possible iterables) of values
        returns     :   list (of possible lists) of ids

        Takes an (possibly nested) iterable and applies id mapping to it recursively, ret
        """
        def map2id(val):
            return self.get_id(val)
        return recursive_map(map2id, iterable)

    def ids2iterables(self, iterable):
        """
        iterable    :   iterable (of possible iterables) of ids
        returns     :   list (of possible lists) of values

        Takes an (possibly nested) iterable and applies id mapping to it recursively
        """

        def map2val(id):
            return self.get_key(id)

        return recursive_map(map2val, iterable)

if __name__ == "__main__":

    r = range(100)
    titles = [
        r[0:10],
        r[5:15],
        r,
        r[0:50],
        r[50:100],
        r[25:75],
        r[10:20],
        r[15:30],
    ]

    from IterableFP import recursive_map

    idgen = IdGenerator()
    titles = recursive_map(str, titles)
    ids = idgen.iterable2ids(titles)
    pass