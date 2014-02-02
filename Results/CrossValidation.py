"""
    Splits a list in to #folds for cross validation
"""
import Metrics
import numpy as np

def __ensure_np_array__(arr):
    if type(arr) != np.ndarray:
        arr = np.array(arr)

    """
    Warning - can cause nested arrays for some algorithms
    e.g. OrderedRule learner. This code is probably to ensure output is column
        and not a row vector (which it does ensure if input has 1 dimension not multiple)
    if len(arr.shape) == 1: #one not multi-dimensional
        arr = np.reshape(arr, (arr.shape[0], 1))
    """
    return arr

def __ensure_column_vector__(arr):
    if type(arr) != np.ndarray:
        raise Exception("Numpy array expected")


def cross_validation_edges(n, folds):
    split = n / folds
    rem = n % folds
    
    #List of lists
    edges = []
    start = 0
    
    for i in range(0,folds):
        size = split
        if i < rem:
            size += 1
        edges.append((start, start + size))
        start += size
    return edges

def cross_validation(lst, folds):    
    edges = cross_validation_edges(len(lst), folds)
    cv = []
    
    for i in range(0,folds):
        l, r = edges[i]
        train  = lst[:l] + lst[r:]
        validn = lst[l:r]
        cv.append((train, validn))
    return cv

def cross_validation_score(x, y, classifier, folds = 10, class_value = 1.0):
    """ Creates #folds in the dataset, and then runs the 
        <classifier> on them, computing the average recall,
        precision and f1 score
    """
    if len(x) != len(y):
        raise Exception("Lists are not the same size")
    
    x = __ensure_np_array__(x)
    y = __ensure_np_array__(y)
    
    edges = cross_validation_edges(len(x), folds)
    recall, precision, f1_score = 0.0, 0.0, 0.0
    
    
    for i in range(folds):
        l,r = edges[i]
        
        #Note these are numpy obj's and cannot be treated as lists
        td_x = np.concatenate((x[:l], x[r:]))                            
        td_y = np.concatenate((y[:l], y[r:]))
        
        vd_x = x[l:r]
        vd_y = y[l:r]
        
        classifier.fit(td_x, td_y)
        pred_y = classifier.predict(vd_x)
        
        r, p, f1 = Metrics.rpf1(vd_y, pred_y, class_value)
        recall    += r
        precision += p
        f1_score  += f1
    
    recall      = recall    / folds
    precision   = precision / folds
    f1_score    = f1_score  / folds
    
    return (recall, precision, f1_score)

def cross_validation_score_generic(x, y, fn_create_classifier, fn_classify, folds = 10, class_value = 1.0, one_fold = False):
    """ Creates #folds in the dataset, and then runs the 
        <classifier> on them, computing the average recall,
        precision and f1 score
        fn_create_classifier : a function that takes a list of training data and returns a classifier
        fn_classifier        : a function that takes a classifier and a list of inputs and returns a list of classifications
        folds                : Number of folds
        class_value          : positive class value
        one_fold             : run for one fold (for quick testing)
    """
    if len(x) != len(y):
        raise Exception("Lists are not the same size")

    npx = __ensure_np_array__(x)
    npy = __ensure_np_array__(y)
    
    edges = cross_validation_edges(len(x), folds)

    td_recall, td_precision, td_f1_score, td_accuracy = 0.0, 0.0, 0.0, 0.0
    vd_recall, vd_precision, vd_f1_score, vd_accuracy = 0.0, 0.0, 0.0, 0.0

    td_tp_ix, td_fp_ix, td_fn_ix, td_tn_ix = [], [], [], []
    vd_tp_ix, vd_fp_ix, vd_fn_ix, vd_tn_ix = [], [], [], []

    for i in range(folds):
        l,r = edges[i]
        
        #Note these are numpy obj's and cannot be treated as lists
        td_x = np.concatenate((npx[:l], npx[r:]))                            
        td_y = np.concatenate((npy[:l], npy[r:]))
        
        vd_x = np.array(npx[l:r])
        vd_y = np.array(npy[l:r])
        
        classifier = fn_create_classifier(td_x, td_y)

        pred_td_y = fn_classify(classifier, td_x)


        td_r, td_p, td_f1, td_a,     tp_ix, fp_ix, fn_ix, tn_ix,  = Metrics.rpf1a_with_indices(td_y, pred_td_y, class_value)
        td_recall    += td_r
        td_precision += td_p
        td_f1_score  += td_f1
        td_accuracy  += td_a

        td_tp_ix.extend(tp_ix)
        td_fp_ix.extend(fp_ix)
        td_fn_ix.extend(fn_ix)
        td_tn_ix.extend(tn_ix)

        pred_vd_y = fn_classify(classifier, vd_x)

        vd_r, vd_p, vd_f1, vd_a, tp_ix, fp_ix, fn_ix, tn_ix = Metrics.rpf1a_with_indices(vd_y, pred_vd_y, class_value)
        vd_recall    += vd_r
        vd_precision += vd_p
        vd_f1_score  += vd_f1
        vd_accuracy  += vd_a

        vd_tp_ix.extend(tp_ix)
        vd_fp_ix.extend(fp_ix)
        vd_fn_ix.extend(fn_ix)
        vd_tn_ix.extend(tn_ix)

        if one_fold:
            folds = 1
            break

    #Compute mean scores across all folds
    
    mean_td_recall      = td_recall    / folds
    mean_td_precision   = td_precision / folds
    mean_td_f1_score    = td_f1_score  / folds
    mean_td_accuracy    = td_accuracy  / folds

    mean_vd_recall      = vd_recall    / folds
    mean_vd_precision   = vd_precision / folds
    mean_vd_f1_score    = vd_f1_score  / folds
    mean_vd_accuracy    = vd_accuracy  / folds
    
    return \
        (   mean_vd_recall, mean_vd_precision, mean_vd_f1_score, mean_vd_accuracy,
            mean_td_recall, mean_td_precision, mean_td_f1_score, mean_td_accuracy,

            # indices for different groupings
            vd_tp_ix, vd_fp_ix, vd_fn_ix, vd_tn_ix,
            td_tp_ix, td_fp_ix, td_fn_ix, td_tn_ix
        )

if __name__ == '__main__':
    
    def prnt(cv):
        for a,b in cv:
            print a,b, len(b)
    
    l = range(0,13)
    cv = cross_validation(l, 3)
    prnt(cv)
       
    print "\n"
    l = range(0,15)
    cv = cross_validation(l, 3)
    prnt(cv)
    
    print "\n"
    l = range(0,16)
    cv = cross_validation(l, 3)
    prnt(cv)
