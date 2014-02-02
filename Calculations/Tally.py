from collections import defaultdict
'''
Created on Feb 1, 2013

@author: simon
'''
def tally(iterbl):
    """ Given a non-unique collection of items
        compute a dictionary containing
        the frequency of each unique item
    """
    tally = defaultdict(int)
    for item in iterbl:
        tally[item] += 1
    return tally