def fuzzy_match(original, feat_wd):
    original = original.lower().strip()
    feat_wd = feat_wd.lower().strip()
    if original == feat_wd:
        #print "\nMatch"
        return True
    if orig[:3] == feat_wd[:3]:
        #print "\n", orig[:3] , feat_wd[:3]
        return True
    a = set(original)
    b = set(feat_wd)
    jaccard = float(len(a.intersection(b))) / float(len(a.union(b)))
    #print "\nJaccard"
    return jaccard >= 0.5

def align_wd_tags(orig, feats):
    if len(orig) < len(feats):
        raise Exception("align_wd_tags() : Original sentence is longer!")

    o_wds,    _        = zip(*orig)
    feat_wds, new_tags = zip(*feats)

    if len(orig) == len(feats):
        return zip(o_wds, new_tags)

    #here orig is longer than feats
    diff = len(orig) - len(feats)
    tagged_wds = []
    feat_offset = 0
    while len(tagged_wds) < len(o_wds):
        i = len(tagged_wds)
        orig_wd = o_wds[i]
        print i, orig_wd

        if i >= len(feats):
            tagged_wds.append((orig_wd, new_tags[-1]))
            continue
        else:
            new_tag_ix = i - feat_offset
            feat_wd = feats[new_tag_ix][0]
            if feat_wd == "INFREQUENT" or feat_wd.isdigit():
                tagged_wds.append((orig_wd, new_tags[new_tag_ix]))
                continue

            new_tagged_wds = []
            found = False
            for j in range(i, i+diff+1):
                new_tagged_wds.append((o_wds[j], new_tags[new_tag_ix]))
                next_orig_wd = o_wds[j]
                if fuzzy_match(next_orig_wd, feat_wd):
                    found = True
                    tagged_wds.extend(new_tagged_wds)
                    feat_offset += len(new_tagged_wds) -1
                    break
            if not found:
                raise Exception("No matching word found for index:%i and processed word:%s" % (i, feat_wd))
    return tagged_wds

if __name__ == "__main__":
    def print_lst(tag_l):
        print "["
        for wd in zip(*tag_l)[0]:
            print "'%s'," % wd
        print "]"

    def print_tags(tags):
        for wd, tag in tags:
            print str((wd, tag))

    orig = map(lambda wd: (wd, set()), "The plankton was causing the coral to be bleached".split(" "))
    # print_lst(orig)
    tagged = [
        #'The',
        'INFREQUENT',
        'was',
        'causing',
        #'the',
        'coral',
        'to',
        'be',
        'bleached',
    ]
    tagged = map(lambda w: (w, set([w])), tagged)
    #tagged
    aligned = align_wd_tags(orig, tagged)

    print "\nOriginal"
    print_tags(orig)

    print "\nPredicted"
    print_tags(tagged)

    print "\nAligned"
    print_tags(aligned)