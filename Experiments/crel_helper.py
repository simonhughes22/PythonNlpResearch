from collections import defaultdict

def get_cr_tags(pr_tagged_essays, tag_essays_test):

    stag_freq = defaultdict(int)
    unique_words = set()
    for essay in pr_tagged_essays:
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_words.add(word)
                for tag in tags:
                    stag_freq[tag] += 1

    # Ensure we include the test essay tags
    for essay in tag_essays_test:
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_words.add(word)
                for tag in tags:
                    stag_freq[tag] += 1

    crel_tags = list((t for t in stag_freq.keys() if ("->" in t) and
                    not "Anaphor" in t and
                    not "other" in t and
                    not "rhetorical" in t and
                    not "factor" in t and
                    1 == 1
                    ))
    regular_tags = set((t for t in stag_freq.keys() if ("->" not in t) and (t[0].isdigit())))
    # regular_tags = set((t for t in stag_freq.keys() if ( "->" not in t) and (t == "explicit" or t[0].isdigit())))
    assert "explicit" not in regular_tags, "explicit should NOT be in the regular tags"
    return crel_tags