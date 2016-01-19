def friendly_tag(tag):
    return tag.replace("Causer:", "").replace("Result:", "")

def cr_sort_key(cr):
    cr = cr.replace("5b", "5.5")
    # _'s last
    if cr[0] == "_":
        return (99999999, cr, cr, cr)
    # Casual second to last, ordered by the order of the cause then the effect
    if "->" in cr:
        cr = friendly_tag(cr)
        a,b = cr.split("->")
        if a.isdigit():
            a = float(a)
        if b.isdigit():
            b = float(b)
        return (9000, a,b, cr)
    # order causer's before results
    elif "Result:" in cr:
        cr = friendly_tag(cr)
        return (-1, float(cr),-1,cr)
    elif "Causer:" in cr:
        cr = friendly_tag(cr)
        return (-2, float(cr),-1,cr)
    else:
        #place regular tags first, numbers ahead of words
        if cr[0].isdigit():
            return (-10, float(cr),-1,cr)
        else:
            return (-10, 9999.9   ,-1,cr.lower())
    return (float(cr.split("->")[0]), cr) if cr.split("->")[0][0].isdigit() else (99999, cr)

# coding=utf-8
def essay_category(s, essay_type):
    essay_type = essay_type.strip().upper()
    if not s or s == "" or s == "nan":
        return 1

    splt = s.strip().split(",")
    splt = filter(lambda s: len(s.strip()) > 0, splt)
    regular = [t.strip() for t in splt if t[0].isdigit()]
    any_causal = [t.strip() for t in splt if "->" in t and (("Causer" in t and "Result" in t) or "C->R" in t)]
    causal = [t.strip() for t in splt if "->" in t and "Causer" in t and "Result" in t]
    if len(regular) == 0 and len(any_causal) == 0:
        return 1
    if len(any_causal) == 0:  # i.e. by this point regular must have some
        return 2  # no causal
    # if only one causal then must be 3
    elif len(any_causal) == 1 or len(causal) == 1:
        return 3

    # Map to Num->Num, e.g. Causer:3->Results:50 becomes 3->5
    # Also map 6 to 16 and 7 to 17 to enforce the relative size relationship

    def map_cb(code):
        return code.replace("6", "16").replace("7", "17")

    def map_sc(code):
        return code.replace("4", "14").replace("5", "15").replace("6", "16").replace("150", "50")

    is_cb = False
    is_sc = False
    if essay_type == "CB":
        is_cb = True
        crels = sorted(map(lambda t: map_cb(t.replace("Causer:", "").replace("Result:", "")).strip(), causal),
                       key=cr_sort_key)
    elif essay_type == "SC":
        is_sc = True
        crels = sorted(map(lambda t: map_sc(t.replace("Causer:", "").replace("Result:", "")).strip(), \
                           causal),
                       key=cr_sort_key)
    else:
        raise Exception("Unrecognized filename")

    un_results = set()
    # For each unique pairwise combination
    for a in crels:
        for b in crels:
            if cr_sort_key(b) >= cr_sort_key(a):  # don't compare each pair twice (a,b) == (b,a)
                break
            # b is always the smaller of the two
            bc, br = b.split("->")
            ac, ar = a.split("->")
            # if result from a is causer for b
            if br.strip() == ac.strip():
                un_results.add((b, a))

    if len(un_results) >= 1:

        # To be a 4 or a 5, at least one relation needs to end in a 50
        joined = ",".join(map(str, un_results))
        # 50 is the universal code for the essay topic
        if "->50" not in joined:
            return 3

        # CB and 6->7->50 ONLY
        if len(un_results) == 1 and is_cb and ("16->17", "17->50") in un_results:
            return 4
        if len(un_results) <= 2 and is_sc:
            # 4->5->6->50
            codes = set("14,15,16,50".split(","))
            un_results_cp = set(un_results)
            for a, b in un_results:
                alhs, arhs = a.split("->")
                blhs, brhs = b.split("->")
                if alhs in codes and arhs in codes and blhs in codes and brhs in codes:
                    un_results_cp.remove((a, b))
            if len(un_results_cp) == 0:
                return 4
        return 5
    else:
        return 3

import pandas as pd
from PandasHelper import group_by

def get_accuracy(fname, essay_type):

    data = pd.read_csv(fname, sep="|")
    data["Concept Codes"] = data["Concept Codes"].astype("str")
    data["Concept Codes"] = data["Concept Codes"].apply(lambda s: "" if s == "nan" else s)
    data["Predictions"] = data["Predictions"].astype("str")
    data["Predictions"] = data["Predictions"].apply(lambda s: "" if s == "nan" else s)

    def concat(lst):
        return ",".join(lst)

    def make_unique(s):
        joined = s
        splt = joined.split(",")
        if len(splt) == 0:
            return ""
        un = set(splt)
        if "" in un:
            un.remove("")
        return ",".join(sorted(un))

    def codes_only(s):
        splt = s.split(",")
        return ",".join([t for t in splt if len(t.strip()) > 0 and t[0].isdigit()])

    def causal_only(s):
        splt = s.split(",")
        causal = ",".join([t for t in splt if len(t.strip()) > 0 and "->" in t and "Causer" in t and "Result" in t])
        return causal.replace("Causer:", "").replace("Result:", "")

    def category(s):
        return essay_category(s, essay_type)

    grpd = group_by(data, "Essay", {"Concept Codes": concat, "Predictions": concat})

    grpd["Concept Codes"] = grpd["Concept Codes"].apply(make_unique)
    grpd["Predictions"] = grpd["Predictions"].apply(make_unique)

    grpd["Ys_codes"] = grpd["Concept Codes"].apply(codes_only)
    grpd["Pred_codes"] = grpd["Predictions"].apply(codes_only)

    grpd["Ys_causal"] = grpd["Concept Codes"].apply(causal_only)
    grpd["Pred_causal"] = grpd["Predictions"].apply(causal_only)
    # Re-order cols
    grpd = grpd[["Essay", "Concept Codes", "Ys_codes", "Ys_causal", "Predictions", "Pred_codes", "Pred_causal"]]

    grpd["Ys_cat"] = grpd["Concept Codes"].apply(category)
    grpd["Pred_cat"] = grpd["Predictions"].apply(category)

    grpd["Diff"] = grpd["Ys_cat"] - grpd["Pred_cat"]
    grpd["Diff"] = grpd["Diff"].abs()

    s = "Essay_Category" + "\n\n"
    s += "Accuracy:" + str(round(len(grpd[grpd["Ys_cat"] == grpd["Pred_cat"]]) / float(len(grpd)), 4)) + "\n"
    s += "Adj:"     + str(round(len(grpd[grpd["Diff"] <= 1]) / float(len(grpd)), 4)) + "\n"
    s += "NumEssays:" + str(len(grpd)) + "\n"
    return s