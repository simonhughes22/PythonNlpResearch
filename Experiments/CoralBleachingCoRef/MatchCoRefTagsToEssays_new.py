# needed for serialization I think
import dill

from CoRefHelper import parse_stanfordnlp_tagged_essays
from FindFiles import find_files
from Settings import Settings

CV_FOLDS = 5
DEV_SPLIT = 0.1

"""
Why does this file exist? This logic is complex, and rather than copy and paste it across notebooks, i want to ensure
there's a commonly well defined (and easily debuggable) function that can be stepped through, and then imported into
the different nb's as needed (or py scripts).

"""

""" Begin Settings """
DATASET = "CoralBleaching"
PARTITION = "Test" # Training | Test
SCAN_LENGTH = 3
""" END Settings """


settings = Settings()
root_folder = settings.data_directory + DATASET + "/Thesis_Dataset/"
merged_predictions_folder = root_folder + "Predictions/CoRef/MergedTags/"

coref_root = root_folder + "CoReference/"
coref_folder = coref_root + PARTITION

# override this so we don't replace INFREQUENT words
#config["min_df"] = 0

if PARTITION.lower() == "training":
    merged_essays_fname =  "merged_essays_train.dill"
elif PARTITION.lower() == "test":
    merged_essays_fname = "merged_essays_test.dill"
else:
    raise Exception("Invalid partition: " + PARTITION)

with open(merged_predictions_folder + merged_essays_fname, "rb+") as f:
    tagged_essays = dill.load(f)

print("{0} training essays loaded".format(len(tagged_essays)))

# map parsed essays to essay name
essay2tagged = {}
for e in tagged_essays:
    essay2tagged[e.name.split(".")[0]] = e

# Load CoRef Parsed Essays
coref_files = find_files(coref_folder, ".*\.tagged")
print("{0} co-ref tagged files loaded".format(len(coref_files)))
assert len(coref_files) == len(tagged_essays)

essay2coref_tagged = parse_stanfordnlp_tagged_essays(coref_files)

# VALIDATE THE SAME SET OF ESSAYS
assert essay2tagged.keys() == essay2coref_tagged.keys()
intersect = set(essay2tagged.keys()).intersection(essay2coref_tagged.keys())
assert len(intersect) == len(essay2tagged.keys())
assert len(essay2tagged.keys()) > 1
assert len(essay2tagged.keys()) == len(essay2coref_tagged.keys())

failed_cnt = 0
COREF_PHRASE = "COREF_PHRASE"

replacements = []
fuzzy_matches = []

updated_essays = []

def map_tagged_words_to_word_ixs(tagged_essay):

    wd2tags = []
    taggedwd2sentixs = {}
    for sent_ix, sent in enumerate(tagged_essay.sentences):
        for wd_ix, (wd, tags) in enumerate(sent):
            taggedwd2sentixs[len(wd2tags)] = (sent_ix, wd_ix)
            if wd == "\'\'":
                wd = "\""
            wd2tags.append((wd, tags))
    return wd2tags, taggedwd2sentixs


def replace_underscore(mention):
    return set(map(lambda s: s.replace("_"," "), mention))

def map_mentions_to_word_ixs(coref_essay):
    #TODO - fix this, it assume one mention per word, but we can have multiple
    wds2coref = []
    mentions = []
    for sent_ix, sent in enumerate(coref_essay):
        current_mentions = set()
        mention_ixs = set()
        for wd_ix, (wd, tag_dict) in enumerate(sent):
            wds2coref.append((wd, tag_dict))
            if COREF_PHRASE not in tag_dict:
                if len(current_mentions) > 0:
                    mentions.append((current_mentions, mention_ixs))
                current_mentions = set()
                mention_ixs = set()
            else:
                phrases = replace_underscore(tag_dict[COREF_PHRASE])
                if phrases != current_mentions and len(current_mentions) > 0:
                    mentions.append((current_mentions, mention_ixs))
                    current_mentions = set()
                    mention_ixs = set()
                current_mentions = phrases
                mention_ixs.add(len(wds2coref) - 1)
        if len(current_mentions) > 0:
            mentions.append((current_mentions, mention_ixs))
    return wds2coref, mentions


def map_words_between_essays(wd2_tags, wds2coref):
    errors = []

    ix_tagd, ix_coref = 0, 0
    ixtagd_2_ixcoref = {}
    ixcoref_2_ixtagd = {}

    while ix_tagd < (len(wd2tags) - 1) and ix_coref < (len(wds2coref) - 1):
        wd_tagd, atags = wd2tags[ix_tagd]
        wd_coref, btag_dict = wds2coref[ix_coref]

        if wd_tagd == wd_coref or wd_tagd == "cannot" and wd_coref == "can":
            ixtagd_2_ixcoref[ix_tagd] = ix_coref
            ixcoref_2_ixtagd[ix_coref] = ix_tagd
            ix_tagd += 1
            ix_coref += 1
        else:
            # look ahead in wds2 for item that matches next a
            found_match = False
            for offseta, (aa, atags) in enumerate(wd2tags[ix_tagd: ix_tagd + 1 + SCAN_LENGTH]):
                for offsetb, (bb, bb_tag_dict) in enumerate(wds2coref[ix_coref:ix_coref + 1 + SCAN_LENGTH]):
                    if aa == bb:
                        if offseta == offsetb:
                            for i in range(ix_tagd, ix_tagd + offseta):
                                if i not in ixtagd_2_ixcoref:
                                    ixtagd_2_ixcoref[i] = i

                        ix_tagd = ix_tagd + offseta
                        ix_coref = ix_coref + offsetb
                        ixtagd_2_ixcoref[ix_tagd] = ix_coref
                        ixcoref_2_ixtagd[ix_coref] = ix_tagd
                        found_match = True
                        break
                if found_match:
                    break
            if not found_match:
                errors.append((ename, wd_tagd, wd_coref, ix_tagd, ix_coref))
                break
    return ixtagd_2_ixcoref, ixcoref_2_ixtagd, errors

for ename, coref_essay in essay2coref_tagged.items():
    assert ename in essay2tagged
    tagged_essay = essay2tagged[ename]

    wd2tags, taggedwd2sentixs = map_tagged_words_to_word_ixs(tagged_essay)
    wds2coref, mentions = map_mentions_to_word_ixs(coref_essay)

    # if len(mentions) == 0:
    #     continue

    ixtagd_2_ixcoref, ixcoref_2_ixtagd, errors = map_words_between_essays(wd2tags, wds2coref)
    if errors:
        # Print errors
        for ename, wd_tagd, wd_coref, ix_tagd, ix_coref in errors:
            failed_cnt += 1
            print("Failed: " + ename, wd_tagd, wd_coref, ix_tagd, ix_coref)

    # replace the corefs/amaphors with their antecedents
    replacement_sent2wdix = {}
    for mention, mention_ixs in mentions:
        first_ix = min(mention_ixs)
        is_fuzzy = False
        if first_ix not in ixcoref_2_ixtagd:
            while first_ix > 0 and first_ix not in ixcoref_2_ixtagd:
                first_ix -= 1
            if first_ix not in ixcoref_2_ixtagd:
                e_first_wd_ix = 0
            # one past last matching index
            else:
                e_first_wd_ix = min(len(wd2tags) - 1, ixcoref_2_ixtagd[first_ix] + 1)
            is_fuzzy = True
        else:
            e_first_wd_ix = ixcoref_2_ixtagd[first_ix]

        last_ix = max(mention_ixs)
        if last_ix not in ixcoref_2_ixtagd:
            while last_ix < len(wds2coref) and last_ix not in ixcoref_2_ixtagd:
                last_ix += 1
            if last_ix not in ixcoref_2_ixtagd:
                e_last_wd_ix = len(wd2tags) - 1
            else:
                e_last_wd_ix = max(0, ixcoref_2_ixtagd[last_ix] - 1)
            is_fuzzy = True
        else:
            e_last_wd_ix = ixcoref_2_ixtagd[last_ix]

        replacement = []
        all_tags = set()
        for e_wd_ix in range(e_first_wd_ix, e_last_wd_ix + 1):
            sent_ix, sent_wd_ix = taggedwd2sentixs[e_wd_ix]
            sentence = tagged_essay.sentences[sent_ix]
            wd, tags = sentence[sent_wd_ix]
            all_tags.update(tags)
            replacement.append((wd, tags))
            if e_wd_ix == e_first_wd_ix:
                replacement_sent2wdix[(sent_ix, sent_wd_ix)] = (mention, all_tags)
            else:
                replacement_sent2wdix[(sent_ix, sent_wd_ix)] = None

        if replacement:
            replacements.append((mention, replacement))
            if is_fuzzy:
                fuzzy_matches.append((mention, replacement))


        if len(replacement) < (len(mention_ixs)):
            print("WARNING", ("|" + mention + "|||").ljust(50), "!" + " ".join(list(zip(*replacement))[0]) + "!!!", len(replacement), len(mention_ixs))

    new_sentences = []
    for sent_ix, sent in enumerate(tagged_essay.sentences):
        changed_ix = -1
        new_sent = []
        for wd_ix, (wd, tags) in enumerate(sent):
            key = (sent_ix, wd_ix)
            if key in replacement_sent2wdix:
                val = replacement_sent2wdix[key]
                if val is not None:
                    changed_ix = wd_ix
                    mention, all_tags = val
                    for mention_wd in mention.split(" "):
                        new_sent.append((mention_wd, all_tags))
            else:
                new_sent.append((wd, tags))
        new_sentences.append(new_sent)
    tagged_essay.sentences = new_sentences
    updated_essays.append(tagged_essay)

print(len(updated_essays), "updated essays")
# with open(coref_root + partition.lower() + "_processed.dill", "wb+") as f:
#     dill.dump(updated_essays, f)

print(failed_cnt)