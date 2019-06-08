from typing import Dict

def build_cb_causal_model():
    cm = dict()
    for i in range(1,5):
        cm[str(i)] = str(i+1)
    cm["5"] = "5b"
    cm["5b"] = "14"
    cm["14"] = "6"
    cm["6"] = "7"
    cm["7"] = "50"
    for i in range(11,14):
        cm[str(i)] = str(i+1)
    return cm

def build_sc_causal_model():
    cm = dict()
    for i in range(1,6):
        cm[str(i)] = str(i+1)
    cm["11"] = "12"
    cm["12"] = "6"
    cm["6"] = "50"
    return cm

def to_numeric_tag(tag: str)->int:
    nums = ""
    for c in tag:
        if c.isdigit():
            nums += c
    return int(nums)

TERMINAL = "50"
def distance_between(a:str, b:str, causal_model: Dict[str, str]):
    # ensure a <= b
    a, b = str(a), str(b)
    cross_over_node = (a == "6" and b in {"12", "14"})
    if cross_over_node or (not cross_over_node and (to_numeric_tag(a) > to_numeric_tag(b))):
        a, b = b, a

    assert a in causal_model or a == TERMINAL, "a is not in the model"
    assert b in causal_model or b == TERMINAL, "b is not in the model"

    return __nodes_between__(a, b, causal_model)

def __nodes_between__(a, b, causal_model):
    nodes_btwn = 0
    current = a
    while current != b and current != TERMINAL:
        current = causal_model[current]
        nodes_btwn += 1
    # couldn't reach the other node
    if b != TERMINAL and current == TERMINAL:
        return -1
    return nodes_btwn

def is_forward_relation(causer, effect, causal_model):
    causer, effect = str(causer), str(effect)
    return __nodes_between__(causer, effect) > 0

def is_starting_node(a):
    a = str(a).strip()
    return a in {"1", "11"}

def is_terminal_node(a):
    a = str(a).strip()
    return a == TERMINAL