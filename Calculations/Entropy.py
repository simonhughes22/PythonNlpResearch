import math
import Tally
'''
Created on Feb 1, 2013

@author: simon
'''
def entropy(lst_labels):
    """
    lst_labels  :   list of something

    Given a list of category labels, computes the entropy of the list
    """
    tally = Tally.tally(lst_labels)
    
    entropy = 0.0
    num_items = float(len(lst_labels))
    for k, cnt in tally.items():
        p = cnt / num_items
        entropy += -1.0 * p * math.log(p, 2)
    return entropy

if __name__ == "__main__":

    def eval_entropy(lst):
        print "Entropy for " + str(lst) + " is " + str(entropy(lst))

    eval_entropy([1,1,1,1,0,0,0,0])
    eval_entropy([1,1,0,0,0,0,0,0])
    eval_entropy([1,0,0,0,0,0,0,0])
    eval_entropy([1,1,1,1,1,1,1,1])
