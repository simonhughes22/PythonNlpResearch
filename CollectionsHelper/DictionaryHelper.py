'''
Created on Jan 21, 2013

@author: Simon
'''

from collections import defaultdict

def tally_items(lst_objects, fn_extract_prop = None, freq_threshold = 0, sort = False):
    """ Given a [list_objects], use [fn_extract_prop] to extract some feature
        then compute a tally for all those features in the dataset and return.
        [freq_threshold] filter out values with a freq below this threshold
        Returns: dict() unless sort = True, then a list of sorted tuples() sorted by freq desc
    """

    if fn_extract_prop == None:
        """ Set to identity fn """
        fn_extract_prop = lambda i:i
    
    tally = defaultdict(int)
    for obj in lst_objects:
        val = fn_extract_prop(obj)
        tally[val] += 1

    # Convert from defaultdict to a dict, to prevent confusion when accessing values
    if freq_threshold > 1:
        retVal = dict([(k,v) for k,v in tally if v >= freq_threshold])
    else:
        retVal = dict(tally.items())
    if sort:
        """ NOTE: Returns a list of tuples, sorted by frequency desc """
        return sort_by_value(retVal, reverse = True)
    """ return a regular dictionary """
    return retVal
    
def __sort_dictionary__(d, fn_sort_by, reverse = False):
    """
    Sorts a dictionary using the fn_sort_by function
    @param d: dictionary to sort
    @param fn_sort_by: function to extract the sort by field
        from the item {key,value} pairing
    @return: a list of tuples {key, value} 
    """
    
    return sorted(d.items(), reverse = reverse, key = fn_sort_by)

def sort_by_key(d, reverse = False):
    """
    Sorts a dictionary by its keys
    @param d: dictionary to sort
    @return: a list of tuples {key, value}
    """
    return __sort_dictionary__(d, lambda item: item[0], reverse)

def sort_by_value(d, reverse = False):
    """
    Sorts a dictionary by its values
    @param d: dictionary to sort
    @return: a list of tuples {key, value}
    """
    return __sort_dictionary__(d, lambda item: item[1], reverse)