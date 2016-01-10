# coding=utf-8
from collections import defaultdict

def get_tag_freq(tagged_essays):
    tag_freq = defaultdict(int)
    for essay in tagged_essays:
        for sentence in essay.sentences:
            un_tags = set()
            for word, tags in sentence:
                for tag in tags:
                    un_tags.add(tag)
            for tag in un_tags:
                tag_freq[tag] += 1
    return tag_freq

def regular_tag(tag):
    return (tag[-1].isdigit() or tag.startswith("Causer") or tag.startswith("Result") or tag.startswith("explicit") or "->" in tag) \
            and not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag or "5b" in tag)
