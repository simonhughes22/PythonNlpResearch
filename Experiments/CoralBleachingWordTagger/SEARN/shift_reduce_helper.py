EMPTY_TAG = "Empty"

ROOT = "root"

SHIFT = "Shift"
REDUCE = "Reduce"
LARC = "LArc"
RARC = "Rarc"
SKIP = "Skip"

def norm_arc(arc):
    return tuple(sorted(arc))

def norm_arcs(arcs):
    return set(map(norm_arc, arcs))

def extract_lr(cr):
    return cr.replace("Causer:", "").replace("Result:", "").split("->")


def normalize_cr(cr):
    pair = tuple(extract_lr(cr))
    return pair

def denormalize_cr(crel):
    l, r = crel
    return "Causer:{l}->Result:{r}".format(l=l, r=r)
