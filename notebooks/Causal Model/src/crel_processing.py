from collections import defaultdict

from BrattEssay import ANAPHORA
from parse_generator import collapse_sent_parse, merge_crel_probs

EMPTY = "Empty"

def to_is_valid_crel(tags):
    filtered = set()
    for t in tags:
        t_lower = t.lower()
        if "rhetorical" in t_lower or "change" in t_lower or "other" in t_lower:
            continue
        if "->" in t and ANAPHORA not in t:
            filtered.add(t)
    return filtered

def get_crel_tags_by_sent(essays_a):
    crels_by_sent = []
    for ea in essays_a:
        for asent in ea.sentences:
            all_atags = set()
            for awd, atags in asent:
                all_atags.update(to_is_valid_crel(atags))
            crels_by_sent.append(all_atags)
    return crels_by_sent

def get_crels(parse):
    crels = set()
    p = parse
    while p:
        if p.relations:
            crels.update(p.relations)
        p = p.parent_action
    return crels


def get_essays2crels(essays, sr_model, top_n, search_mode_max_prob=False):
    trainessay2probs = defaultdict(list)
    for eix, essay in enumerate(essays):
        crel2probs = defaultdict(list)
        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            predicted_tags = essay.pred_tagged_sentences[sent_ix]
            unq_ptags = set([t for t in predicted_tags if t != EMPTY])
            if len(unq_ptags) >= 2:
                pred_parses = sr_model.generate_all_potential_parses_for_sentence(
                    tagged_sentence=taggged_sentence, predicted_tags=predicted_tags, top_n=top_n,
                    search_mode_max_prob=search_mode_max_prob)
                cr2p = collapse_sent_parse(pred_parses)
                merge_crel_probs(crel2probs, cr2p)

        if len(crel2probs) > 0:
            trainessay2probs[essay.name] = dict(crel2probs)
        else:
            trainessay2probs[essay.name] = dict()
    return trainessay2probs


def essay_to_crels_cv(cv_folds, models, top_n, search_mode_max_prob=False):
    essay2crelprobs = defaultdict(list)
    assert len(cv_folds) == len(models)
    for (train, test), mdl in zip(cv_folds, models):
        test2probs = get_essays2crels(test, mdl, top_n, search_mode_max_prob)
        for k,v in test2probs.items():
            assert k not in essay2crelprobs
            essay2crelprobs[k] = v
    return essay2crelprobs