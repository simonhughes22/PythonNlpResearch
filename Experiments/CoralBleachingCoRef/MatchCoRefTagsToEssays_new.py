# needed for serialization I think
from collections import defaultdict

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

def map_tagged_words_to_word_ixs(tagged_essay):

    tagged_wds = []
    taggedwd2sentixs = {}
    for sent_ix, sent in enumerate(tagged_essay.sentences):
        for wd_ix, (wd, tags) in enumerate(sent):
            taggedwd2sentixs[len(tagged_wds)] = (sent_ix, wd_ix)
            if wd == "\'\'":
                wd = "\""
            tagged_wds.append(wd)
    return tagged_wds, taggedwd2sentixs

def replace_underscore(mention):
    return set(map(lambda s: s.replace("_"," "), mention))

def map_mentions_to_word_ixs(coref_essay, keys):
    #TODO - fix this, it assume one mention per word, but we can have multiple
    coref_wd2_tags = []
    coref_ids_2_wd_ixs = defaultdict(list) # maps a coref id to a list of set of ixs
    for sent_ix, sent in enumerate(coref_essay):
        current_coref_ids = set()
        for wd_ix, (wd, tag_dict) in enumerate(sent):
            coref_wd2_tags.append((wd,tag_dict))

            wd_coref_ids = set()
            for k in keys:
                wd_coref_ids.update(tag_dict[k])

            for cref_id in wd_coref_ids:
                prev_ixs = coref_ids_2_wd_ixs[cref_id]
                # continuation of existing sequence
                wd_essay_ix = len(coref_wd2_tags)-1
                if cref_id in current_coref_ids:
                    if len(prev_ixs) == 0:
                        prev_ixs.append(set())
                    prev_ixs[-1].add(wd_essay_ix)
                else:
                    # else create a new set and add it
                    prev_ixs.append({wd_essay_ix})

            current_coref_ids = wd_coref_ids
    return coref_wd2_tags, coref_ids_2_wd_ixs


def map_words_between_essays(tagged_wds, coref_wd2_tags):
    errors = []

    ix_tagd, ix_coref = 0, 0
    ixtagd_2_ixcoref = {}
    ixcoref_2_ixtagd = {}

    while ix_tagd < (len(tagged_wds) - 1) and ix_coref < (len(coref_wd2_tags) - 1):
        wd_tagd = tagged_wds[ix_tagd]
        wd_coref, btag_dict = coref_wd2_tags[ix_coref]

        if wd_tagd == wd_coref or wd_tagd == "cannot" and wd_coref == "can":
            ixtagd_2_ixcoref[ix_tagd] = ix_coref
            ixcoref_2_ixtagd[ix_coref] = ix_tagd
            ix_tagd += 1
            ix_coref += 1
        else:
            # look ahead in wds2 for item that matches next a
            found_match = False
            for offseta, aa in enumerate(tagged_wds[ix_tagd: ix_tagd + 1 + SCAN_LENGTH]):
                for offsetb, (bb, _) in enumerate(coref_wd2_tags[ix_coref:ix_coref + 1 + SCAN_LENGTH]):
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

def print_errors(errors):
    # Print errors
    for ename, wd_tagd, wd_coref, ix_tagd, ix_coref in errors:
        print("Failed: " + ename, wd_tagd, wd_coref, ix_tagd, ix_coref)

updated_essays = []
for ename, coref_essay in essay2coref_tagged.items():
    assert ename in essay2tagged
    tagged_essay = essay2tagged[ename]

    tagged_wds, taggedwd2sentixs = map_tagged_words_to_word_ixs(tagged_essay)

    coref_wd2_tags, coref_ids_2_wd_ixs = map_mentions_to_word_ixs(coref_essay)

    ixtagd_2_ixcoref, ixcoref_2_ixtagd, errors = map_words_between_essays(tagged_wds, coref_wd2_tags)
    if errors:
        failed_cnt += len(errors)
        print_errors(errors)

    # replace the corefs/amaphors with their antecedents

    sent_wdix_2_corefids = defaultdict(set)
    for coref_id, list_wd_ix_seq in coref_ids_2_wd_ixs.items():
        for wd_ixs in list_wd_ix_seq:

            first_ix = min(wd_ixs)
            is_fuzzy = False
            if first_ix not in ixcoref_2_ixtagd:
                while first_ix > 0 and first_ix not in ixcoref_2_ixtagd:
                    first_ix -= 1
                if first_ix not in ixcoref_2_ixtagd:
                    e_first_wd_ix = 0
                # one past last matching index
                else:
                    e_first_wd_ix = min(len(tagged_wds) - 1, ixcoref_2_ixtagd[first_ix] + 1)
                is_fuzzy = True
            else:
                e_first_wd_ix = ixcoref_2_ixtagd[first_ix]

            last_ix = max(wd_ixs)
            if last_ix not in ixcoref_2_ixtagd:
                while last_ix < len(coref_wd2_tags) and last_ix not in ixcoref_2_ixtagd:
                    last_ix += 1
                if last_ix not in ixcoref_2_ixtagd:
                    e_last_wd_ix = len(tagged_wds) - 1
                else:
                    e_last_wd_ix = max(0, ixcoref_2_ixtagd[last_ix] - 1)
                is_fuzzy = True
            else:
                e_last_wd_ix = ixcoref_2_ixtagd[last_ix]

            for e_wd_ix in range(e_first_wd_ix, e_last_wd_ix + 1):
                sent_ix, sent_wd_ix = taggedwd2sentixs[e_wd_ix]
                sentence = tagged_essay.sentences[sent_ix]
                wd, tags = sentence[sent_wd_ix]

                sent_wdix_2_corefids[(sent_ix, sent_wd_ix)].add(coref_id)

    predicted_corefids_sentences = []
    for sent_ix, sent in enumerate(tagged_essay.sentences):
        changed_ix = -1

        predicted_coref_ids_wds = []
        predicted_corefids_sentences.append(predicted_coref_ids_wds)

        for wd_ix in range(len((sent))):
            wd_coref_ids = sent_wdix_2_corefids[(sent_ix, wd_ix)]
            predicted_coref_ids_wds.append(wd_coref_ids)

    tagged_essay.predicted_corefids = predicted_corefids_sentences
    updated_essays.append(tagged_essay)

print(len(updated_essays), "updated essays")
# with open(coref_root + partition.lower() + "_processed.dill", "wb+") as f:
#     dill.dump(updated_essays, f)

print(failed_cnt)