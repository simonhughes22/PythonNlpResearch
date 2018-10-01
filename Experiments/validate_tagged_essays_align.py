from typing import List
import BrattEssay
import processessays

def essays_2_hash_map(essays):
    lu = {}
    for e in essays:
        lu[e.name] = e
    return lu

# checks the number of words and sentences are the same for 2 sets of essays
def validate_tagged_essays(
        essays_a, essays_b,
        validate_tags: bool = True, validate_tag_subset=False, validate_words: bool = True) -> None:

    if validate_tags and validate_tag_subset:
        raise Exception("Validing tags and tag subset, which is redundant")

    # make sure obj is not the same
    assert essays_a != essays_b
    print("Validating", len(essays_a), "essays")
    assert len(essays_a) == len(essays_b), "Lens don't match"

    a_hmap = essays_a
    b_hmap = essays_b

    if type(essays_a) == list:
        a_hmap = essays_2_hash_map(essays_a)

    if type(essays_b) == list:
        b_hmap = essays_2_hash_map(essays_b)

    # same essays?
    assert a_hmap.keys() == b_hmap.keys()
    intersect = set(a_hmap.keys()).intersection(b_hmap.keys())
    assert len(intersect) == len(a_hmap.keys())
    assert len(a_hmap.keys()) > 1
    assert len(a_hmap.keys()) == len(b_hmap.keys())

    word_misses = 0
    tag_misses = 0

    for key, a_essay in a_hmap.items():
        b_essay = b_hmap[key]
        # assert NOT the same obj ref
        assert a_essay != b_essay, "Essay objects ARE the same (bug in validation?)"
        assert len(a_essay.sentences) == len(b_essay.sentences), "Number of sentences differ for essay name '{name}'".format(name=key)
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

                if validate_words:
                    assert a_wd == b_wd, \
                        "Words don't match: '{a}' - '{b}', Esssay: {essay} Sent Ix: {i}".format(
                            a=a_wd, b=b_wd, essay=key, i=i)

                # SH - Make conditional, as untagged essays contain new anaphora tags
                if a_tags != b_tags:
                    tag_misses +=1

                if validate_tags:
                    assert a_tags == b_tags, \
                        "Tags don't match: '{a}' - '{b}', Esssay: {essay} Sent Ix: {i}".format(
                            a=str(a_tags), b=str(b_tags), essay=key, i=i)

                elif validate_tag_subset:
                    intersectn = a_tags.intersection(b_tags)
                    # smaller set should match intersection i.e. be a subset of larger one
                    # will only differ due to new anaphora tags
                    if len(b_tags) <= len(a_tags):
                        assert intersectn == b_tags
                    else:
                        assert intersectn == a_tags

    print("Word miss-matches: {misses}".format(misses=word_misses))
    print("Tag miss-matches:  {misses}".format(misses=tag_misses))
    print("Validation Passed")
    return None