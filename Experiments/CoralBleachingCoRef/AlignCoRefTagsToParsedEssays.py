from Decorators import memoize_to_disk
from FindFiles import find_files
from Settings import Settings
from load_data import load_process_essays
from window_based_tagger_config import get_config
import dill

CV_FOLDS = 5
DEV_SPLIT = 0.1

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
partition = "Training" # Training | Test
target_folder = root_folder + partition + "/"
processed_essay_filename_prefix =  root_folder + "Pickled/essays_proc_pickled_"

config = get_config(target_folder)
# override this so we don't replace INFREQUENT words
#config["min_df"] = 0

mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays(**config)
print("{0} essays loaded".format(len(tagged_essays)))
# map parsed essays to essay name
essay2tagged = {}
for e in tagged_essays:
    essay2tagged[e.name.split(".")[0]] = e

coref_root = root_folder + "CoReference/"
coref_folder = coref_root + partition

coref_files = find_files(coref_folder, ".*\.tagged")
print("{0} co-ref tagged files loaded".format(len(coref_files)))
assert len(coref_files) == len(tagged_essays)


def parse_stanfordnlp_tagged_essays(coref_files):
    DELIM = "->"
    DELIM_TAG = "|||"

    essay2tagged = {}
    for fname in coref_files:
        with open(fname) as f:
            lines = f.readlines()

        tagged_lines = []
        for line in lines:
            tagged_words = []
            line = line.strip()
            wds = []
            for t_token in line.split(" "):
                ##print(t_token)

                word, tags = t_token.split(DELIM)
                if word == "-lrb-":
                    word = "("
                if word == "-rrb-":
                    word = ")"
                if word == "\'\'":
                    word = "\""
                # if word == "not" and len(wds) > 0 and wds[-1] == "can":
                #     last_wd, tag_dict = tagged_words[-1]
                #     tagged_words[-1] = ("cannot", tag_dict)
                #     wds[-1] = "cannot"
                #     continue

                wds.append(word)
                tag_dict = {}
                for tag in tags.split(DELIM_TAG):
                    if not tag:
                        continue
                    splt = tag.split(":")
                    if len(splt) == 2:
                        key, val = splt
                        tag_dict[key] = val
                    else:
                        raise Exception("Error")
                if word == "...":
                    tagged_words.append((".", tag_dict))
                    tagged_words.append((".", tag_dict))
                    tagged_words.append((".", tag_dict))
                else:
                    tagged_words.append((word, tag_dict))
            tagged_lines.append(tagged_words)
        essay2tagged[fname.split("/")[-1].split(".")[0]] = tagged_lines
    return essay2tagged

essay2coref_tagged = parse_stanfordnlp_tagged_essays(coref_files)
assert len(essay2tagged) == len(essay2coref_tagged)

failed_cnt = 0
COREF_PHRASE = "COREF_PHRASE"
SCAN_LENGTH = 3

replacements = []
fuzzy_matches = []

updated_essays = []
for ename, coref_essay in essay2coref_tagged.items():
    assert ename in essay2tagged
    tagged_essay = essay2tagged[ename]

    wds_tagd = []
    taggedwd2sentixs = {}
    replacement_sent2wdix = {}
    for sent_ix, sent in enumerate(tagged_essay.sentences):
        for wd_ix, (wd, tags) in enumerate(sent):
            taggedwd2sentixs[len(wds_tagd)] = (sent_ix, wd_ix)
            if wd == "\'\'":
                wd = "\""
            wds_tagd.append((wd, tags))

    wds_coref = []
    mentions = []
    for sent_ix, sent in enumerate(coref_essay):
        current_mention = ""
        mention_ixs = set()
        for wd_ix, (wd, tag_dict) in enumerate(sent):
            wds_coref.append((wd, tag_dict))
            if COREF_PHRASE not in tag_dict:
                if current_mention != "":
                    mentions.append((current_mention, mention_ixs))
                current_mention = ""
                mention_ixs = set()
            else:
                phrase = tag_dict[COREF_PHRASE].replace("_", " ")
                if phrase != current_mention and current_mention != "":
                    mentions.append((current_mention, mention_ixs))
                    current_mention = ""
                    mention_ixs = set()
                current_mention = phrase
                mention_ixs.add(len(wds_coref) - 1)
        if current_mention != "":
            mentions.append((current_mention, mention_ixs))

    # if len(mentions) == 0:
    #     continue

    ix_tagd, ix_coref = 0, 0
    ixtagd_2_ixcoref = {}
    ixcoref_2_ixtagd = {}

    while ix_tagd < (len(wds_tagd) - 1) and ix_coref < (len(wds_coref) - 1):
        wd_tagd, atags = wds_tagd[ix_tagd]
        wd_coref, btag_dict = wds_coref[ix_coref]

        if wd_tagd == wd_coref or wd_tagd == "cannot" and wd_coref == "can":
            ixtagd_2_ixcoref[ix_tagd]  = ix_coref
            ixcoref_2_ixtagd[ix_coref] = ix_tagd
            ix_tagd  += 1
            ix_coref += 1
        else:
            # look ahead in wds2 for item that matches next a
            found_match = False
            for offseta, (aa, atags) in enumerate(wds_tagd[ix_tagd: ix_tagd + 1 + SCAN_LENGTH]):
                for offsetb, (bb, bb_tag_dict) in enumerate(wds_coref[ix_coref:ix_coref + 1 + SCAN_LENGTH]):
                    if aa == bb:
                        if offseta == offsetb:
                            for i in range(ix_tagd, ix_tagd + offseta):
                                if i not in ixtagd_2_ixcoref:
                                    ixtagd_2_ixcoref[i] = i

                        ix_tagd  = ix_tagd + offseta
                        ix_coref = ix_coref + offsetb
                        ixtagd_2_ixcoref[ix_tagd] = ix_coref
                        ixcoref_2_ixtagd[ix_coref] = ix_tagd
                        found_match = True
                        break
                if found_match:
                    break
            if not found_match:
                print("Failed: " + ename, wd_tagd, wd_coref, ix_tagd, len(wds_tagd), ix_coref, len(wds_coref))
                failed_cnt += 1
                break

    # replace the corefs/amaphors with their antecedents
    for mention, ixs in mentions:
        first_ix = min(ixs)
        is_fuzzy = False
        if first_ix not in ixcoref_2_ixtagd:
            while first_ix > 0 and first_ix not in ixcoref_2_ixtagd:
                first_ix -= 1
            if first_ix not in ixcoref_2_ixtagd:
                e_first_wd_ix = 0
            # one past last matching index
            else:
                e_first_wd_ix = min(len(wds_tagd) - 1, ixcoref_2_ixtagd[first_ix] + 1)
            is_fuzzy = True
        else:
            e_first_wd_ix = ixcoref_2_ixtagd[first_ix]

        last_ix = max(ixs)
        if last_ix not in ixcoref_2_ixtagd:
            while last_ix < len(wds_coref) and last_ix not in ixcoref_2_ixtagd:
                last_ix += 1
            if last_ix not in ixcoref_2_ixtagd:
                e_last_wd_ix = len(wds_tagd) - 1
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


        if len(replacement) < (len(ixs)):
            print("WARNING", ("|" + mention + "|||").ljust(50), "!"+ " ".join(list(zip(*replacement))[0]) + "!!!" , len(replacement), len(ixs))

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
with open(coref_root + partition.lower() + "_processed.dill", "wb+") as f:
    dill.dump(updated_essays, f)

print(failed_cnt)