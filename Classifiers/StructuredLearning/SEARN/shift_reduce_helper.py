EMPTY_TAG = "Empty"

# Special tags
ROOT = "root"
SENT = "<SENT>"

# Actions
SHIFT = "Shift"
REDUCE = "Reduce"
LARC = "LArc"
RARC = "Rarc"
SKIP = "Skip"

PARSE_ACTIONS = [
    SHIFT,
    REDUCE,
    LARC,
    RARC,
    SKIP
]

CAUSE_EFFECT = "CAUSE_EFFECT"
EFFECT_CAUSE = "EFFECT_CAUSE"
CAUSE_AND_EFFECT = "CAUSE_AND_EFFECT"
REJECT = "REJECT"  # Not a CREL

CREL_ACTIONS = [
    CAUSE_EFFECT,
    EFFECT_CAUSE,
    CAUSE_AND_EFFECT,
    REJECT
]

def norm_arc(arc):
    return tuple(sorted(arc, key=lambda tpl: ("", 0) if tpl is None else (tpl[0],tpl[1])))

def norm_arcs(arcs):
    return set(map(norm_arc, arcs))

def extract_lr(cr):
    return cr.replace("Causer:", "").replace("Result:", "").split("->")

def normalize(code):
    return code.replace("Causer:","").replace("Result:","")

def normalize_cr(cr):
    return tuple(normalize(cr).split("->"))

def denormalize_cr(crel):
    l, r = crel
    return "Causer:{l}->Result:{r}".format(l=l, r=r)

def allowed_action(action, tos):
    return not(tos == ROOT and action in (REDUCE, LARC, RARC))

