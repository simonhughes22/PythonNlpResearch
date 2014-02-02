__author__ = 'simon.hughes'

def print_tree(s):
    indent = "    "

    pr_val = ""
    level = -1
    for ch in s:
        if ch == "(":
            if pr_val.strip() != "":
                print pr_val
            level += 1
            pr_val = (indent * level) + "("
        elif ch == ")":
            pr_val += ")"
            print pr_val
            pr_val = (indent * (level))
            level -= 1
        else:
            pr_val += ch
    if len(pr_val) > 0:
        print pr_val

con_codes = """ ((T27 M (0 6) "Humans" NIL NIL 0 1 1 NIL NIL) (T10 50 (23 53) "changes in global temperatures" NIL NIL 0 5 8 NIL NIL) (T2 EXPLICIT (54 66) "in many way." NIL NIL 0 9 12 NIL NIL) (T1 EXPLICIT (11 18) "causing" NIL NIL 0 3 3 NIL NIL)) """
dep_parse = """ ((ROOT (S (NP (NNS HUMANS)) (VP (VBP ARE) (VP (VBG CAUSING) (NP (DT THE) (NNS CHANGES)) (PP (IN IN) (NP (NP (JJ GLOBAL) (NNS TEMPERATURES)) (PP (IN IN) (NP (JJ MANY) (NN WAY))))))) (|.| |.|))) (NSUBJ ("causing" 3) ("Humans" 1)) (AUX ("causing" 3) ("are" 2)) (ROOT ("ROOT" 0) ("causing" 3)) (DET ("changes" 5) ("the" 4)) (DOBJ ("causing" 3) ("changes" 5)) (AMOD ("temperatures" 8) ("global" 7)) (PREP_IN ("causing" 3) ("temperatures" 8)) (AMOD ("way" 11) ("many" 10)) (PREP_IN ("temperatures" 8) ("way" 11)))"""

#print_tree("(root (child a (grandchild b c))(child b))")
print_tree(con_codes)
print_tree(dep_parse)