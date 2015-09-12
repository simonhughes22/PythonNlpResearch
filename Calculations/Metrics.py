'''
Created on Mar 30, 2013

@author: Simon
'''


def __tally_results__(expected, actual, class_value):
    """ Counts the number of true positives, false positives and false negatives
    """

    if len(actual) != len(expected):
        raise Exception("Both list must be the same size, actual - %i expected - %i" % ( len(actual), len(expected)) )
    
    tp = 0.0
    fp = 0.0
    fn = 0.0
    
    for i in range(0, len(actual)):
        act = actual[i]
        exp = expected[i]
        if act == exp:
            if exp == class_value:
                tp += 1.0
        else:
            if exp == class_value:
                fn += 1.0
            else:
                fp += 1.0
    
    return (tp, fp, fn)

def __tally_results_with_indices__(expected, actual, class_value):
    """ Counts the number of true positives, false positives and false negatives,
        AND returns the indices into categories classifications
    """

    if len(actual) != len(expected):
        raise Exception("Both list must be the same size")

    tp = 0.0
    fp = 0.0
    fn = 0.0

    tp_ix = []
    tn_ix = []
    fp_ix = []
    fn_ix = []

    for i in range(0, len(actual)):
        act = actual[i]
        exp = expected[i]
        if act == exp:
            # is positive
            if exp == class_value:
                tp += 1.0
                tp_ix.append(i)
            else:
                tn_ix.append(i)
        else:
            if exp == class_value:
                fn += 1.0
                fn_ix.append(i)
            else:
                fp += 1.0
                fp_ix.append(i)

    return (tp, fp, fn,     tp_ix, fp_ix, fn_ix, tn_ix)

def __precision__(tp, fp, fn):
    if tp + fp <= 0:
        return 0.0
    return tp / (tp + fp)

def precision(expected, actual, class_value = 1):
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    return __precision__(tp, fp, fn)

def __recall__(tp, fp, fn):
    if tp + fn <= 0:
        return 0.0
    return tp / (tp + fn)

def recall(expected, actual, class_value = 1):
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    return __recall__(tp, fp, fn)
    
def __f_beta__(r, p, beta):
    if r + p <= 0.0:
        return 0.0
    beta_squared = beta * beta
    #Harmonic mean
    return ((1.0 + beta_squared) * r * p) / (beta_squared * (r + p))

def f_beta(expected, actual, class_value = 1, beta = 1.0):
    
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    r = __recall__(tp, fp, fn)
    p = __precision__(tp, fp, fn)
    return __f_beta__(r, p, beta)

def f1_score(expected, actual, class_value = 1):
    return f_beta(expected, actual, class_value, 1.0)

def __accuracy__(tp, fp, fn, actual):
    ln = float(len(actual))
    return (ln - (fp + fn)) / ln 

def accuracy(expected, actual, class_value = 1):
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    return __accuracy__(tp, fp, fn, actual)

def rpf1(expected, actual, class_value = 1):
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    
    r = __recall__(tp, fp, fn)
    p = __precision__(tp, fp, fn)
    f1 = __f_beta__(r, p, 1.0)
    return (r,p,f1)

def rpf1a(expected, actual, class_value = 1):
    
    tp, fp, fn = __tally_results__(expected, actual, class_value)
    
    r  = __recall__(tp, fp, fn)
    p  = __precision__(tp, fp, fn)
    f1 = __f_beta__(r, p, 1.0)
    a  = __accuracy__(tp, fp, fn, actual)
    
    return (r,p,f1,a)

def rpf1a_with_indices(expected, actual, class_value = 1):

    tp, fp, fn, tp_ix, fp_ix, fn_ix, tn_ix = __tally_results_with_indices__(expected, actual, class_value)

    r  = __recall__(tp, fp, fn)
    p  = __precision__(tp, fp, fn)
    f1 = __f_beta__(r, p, 1.0)
    a  = __accuracy__(tp, fp, fn, actual)

    return (r,p,f1,a, tp_ix, fp_ix, fn_ix, tn_ix)