import ListHelper

def partition(lst, xs, codes):
    d = {}
    for code in codes:
        indices = xs.indices_for(code)
        d[code] = ListHelper.filter_list_by_index(lst, indices)

    return d