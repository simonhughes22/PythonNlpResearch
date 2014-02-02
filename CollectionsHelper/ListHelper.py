def filter_list_by_index(lst, indices):
    """ Filters a list by the set of indices passed in """
    if not type(lst) == type(list()):
        raise Exception("Expected argument type - list but received: " + str(type(lst)))
    
    # Make hashable for speed 
    count = len(lst)

    return [lst[i] 
            for i in indices
            if (i < 0 and abs(i) <= count) or (i >= 0 and i <= count)]
