
def vector_space_to_dict_list(distance_matrix, id2Word, value_extractor = lambda item : item, fn_dict_creator = lambda : dict()):
    """
        Takes a distance_matrix of the form [[(id,word)]]
        and converts to a lst of dict's {"word", freq}
        Params:
            1. List of lst of tuples
            2. id2Word get sim dictionary mapping id's to words
            3. function to process the frequency (defaults to identity)
    """
    lst = []
    for vector in distance_matrix:
        d = fn_dict_creator()
        for tpl in vector:
            d[id2Word[tpl[0]]] = value_extractor(tpl[1])
        lst.append(d)

    return lst

def to_binary(item):
        if item > 0.0:
            return 1
        return 0

def get_svm_val(x):
    if x <= 0:
        return -1
    return 1
