from collections import defaultdict

def get_tag_freq(train_tagged_essays, tag_essays_test):
    stag_freq = defaultdict(int)
    unique_words = set()
    for essay in train_tagged_essays:
        for sentence in essay.sentences:
            for word, tags in sentence:
                for tag in tags:
                    stag_freq[tag] += 1

    # Ensure we include the test essay tags
    for essay in tag_essays_test:
        for sentence in essay.sentences:
            for word, tags in sentence:
                for tag in tags:
                    stag_freq[tag] += 1
    return stag_freq

def get_unique_words(train_tagged_essays, tag_essays_test):
    unique_words = set()
    for essay in train_tagged_essays:
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_words.add(word)

    # Ensure we include the test essay tags
    for essay in tag_essays_test:
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_words.add(word)
    return unique_words

def get_regular_tags(train_tagged_essays, tag_essays_test):
    tag_freq = get_tag_freq(train_tagged_essays, tag_essays_test)
    regular_tags = set((t for t in tag_freq.keys() if ("->" not in t) and (t[0].isdigit())))
    assert "explicit" not in regular_tags, "explicit should NOT be in the regular tags"
    return regular_tags

def get_cr_tags(train_tagged_essays, tag_essays_test):

    tag_freq = get_tag_freq(train_tagged_essays, tag_essays_test)
    crel_tags = list((t for t in tag_freq.keys() if ("->" in t) and
                    not "Anaphor" in t and
                    not "other" in t and
                    not "rhetorical" in t and
                    not "factor" in t and
                    1 == 1
                    ))
    return crel_tags