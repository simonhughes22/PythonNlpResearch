from typing import List

import BrattEssay


def essays_2_hash_map(essays: List[BrattEssay.Essay]):
    lu = {}
    for e in essays:
        lu[e.name] = e
    return lu

# checks the number of words and sentences are the same for 2 sets of essays
def validate_tagged_essays(
        essays_a: List[BrattEssay.Essay], essays_b: List[BrattEssay.Essay], tags_should_match:bool =True)->None:

    # make sure obj is not the same
    assert essays_a != essays_b
    print("Validating", len(essays_a), "essays")
    assert len(essays_a) == len(essays_b), "Lens don't match"

    a_hmap = essays_2_hash_map(essays_a)
    b_hmap = essays_2_hash_map(essays_b)

    # same essays?
    assert a_hmap.keys() == b_hmap.keys()
    intersect = set(a_hmap.keys()).intersection(b_hmap.keys())
    assert len(intersect) == len(a_hmap.keys())
    assert len(a_hmap.keys()) > 1
    assert len(a_hmap.keys()) == len(b_hmap.keys())

    word_misses = 0

    for key, a_essay in a_hmap.items():
        b_essay = b_hmap[key]
        # assert NOT the same obj ref
        assert a_essay != b_essay
        assert len(a_essay.sentences) == len(b_essay.sentences)
        assert len(a_essay.sentences) > 0
        assert len(b_essay.sentences) > 0
        for i in range(len(a_essay.sentences)):
            a_sent = a_essay.sentences[i]
            b_sent = b_essay.sentences[i]
            # the same lists?
            # assert a_sent == b_sent
            assert len(a_sent) == len(b_sent)
            if not len(a_sent) == len(b_sent):
                print(key, "\tsent-ix:", i, "lens", len(a_sent), len(b_sent))
            for wd_ix, (a_wd, a_tags) in enumerate(a_sent):
                b_wd, b_tags = b_sent[wd_ix]
                if a_wd != b_wd:
                    word_misses += 1
                assert a_wd == b_wd, \
                    "Words don't match: '{a}' - '{b}', Esssay: {essay} Sent Ix: {i}".format(
                        a=a_wd, b=b_wd, essay=key, i=i)

                # SH - Make conditional, as untagged essays contain new anaphora tags
                if tags_should_match:
                    assert a_tags == b_tags, \
                        "Tags don't match: '{a}' - '{b}', Esssay: {essay} Sent Ix: {i}".format(
                            a=str(a_tags), b=str(b_tags), essay=key, i=i)
                else:
                    intersectn = a_tags.intersection(b_tags)
                    # smaller set should match intersection i.e. be a subset of larger one
                    # will only differ due to new anaphora tags
                    if len(b_tags) <= len(a_tags):
                        assert intersectn == b_tags
                    else:
                        assert intersectn == a_tags

    if word_misses:
        print("Word miss-matches: ", word_misses)
    print("Validation Passed")
    return None