import dill
import pandas as pd

from Settings import Settings
from collections import defaultdict
from BrattEssay import ANAPHORA
from CoRefHelper import EMPTY

from results_common import get_essays, validate_essays, tally_essay_attributes
from process_essays_coref import get_coref_processed_essays
from process_essays_coref import build_segmented_chain

from metrics import get_metrics_raw
from results_procesor import is_a_regular_code

DATASET = "CoralBleaching" # CoralBleaching | SkinCancer

settings = Settings()
root_folder = settings.data_directory + DATASET + "/Thesis_Dataset/"
stanford_coref_predictions_folder = root_folder + "CoReference/"
berkeley_coref_predictions_folder = root_folder + "CoReference/Berkeley/"
coref_predictions_folder = berkeley_coref_predictions_folder
print("CoRef Data: ", stanford_coref_predictions_folder)


training_essays = get_essays(coref_predictions_folder, "Training")
test_essays = get_essays(coref_predictions_folder, "Test")
all_essays = training_essays + test_essays


# ner_tally = tally_essay_attributes(all_essays, attribute_name="pred_ner_tags_sentences")
pos_tally = tally_essay_attributes(all_essays, attribute_name="pred_pos_tags_sentences")

cc_tally = defaultdict(int)
cr_tally = defaultdict(int)
reg_tally = defaultdict(int)
for e in all_essays:
    for sent in e.sentences:
        for wd, tags in sent:
            for t in tags:
                if is_a_regular_code(t):
                    reg_tally[t] += 1
                if ANAPHORA in t and "other" not in t:
                    if "->" in t:
                        cr_tally[t] += 1
                    elif "Anaphor:[" in t:
                        cc_tally[t] += 1

reg_tags = sorted(reg_tally.keys())
all_ana_tags = sorted(cc_tally.keys())
assert len(reg_tags) == len(all_ana_tags)


DESIRED_CHAIN_LEN = 4
MIN_ANA_TAGS = 3

matching_essays = []
for e in all_essays:
    has_anaphora_tags = False
    tally_ana_tags = defaultdict(int)
    for sent in e.sentences:
        for wd,tag in sent:
            for t in tag:
                if "anaphor" in t.lower():
                    tally_ana_tags[t] +=1
    
    if len(tally_ana_tags) == 0:
        continue
    max_ana_tags = max(tally_ana_tags.values())
    if max_ana_tags < MIN_ANA_TAGS:
        continue
        
    corefid2chain = build_segmented_chain(e)
    max_len = 0
    for corefid, segmented_chain in corefid2chain.items():
        chain_len = len(segmented_chain)
        if chain_len == DESIRED_CHAIN_LEN:
            matching_essays.append(e)

NEAREST_REF_ONLY = "Nearest reference"
MAX_ANA_PHRASE = "Max ana phrase"
MAX_CHAIN_PHRASE = "Max chain phrase"
POS_ANA_FLTR = "POS ana filter"
POS_CHAIN_FLTR = "Pos chain filter"

def process_sort_results(df_results):
    df_disp = df_results[["f1_score","precision","recall", 
                          NEAREST_REF_ONLY, MAX_ANA_PHRASE, MAX_CHAIN_PHRASE, POS_ANA_FLTR, POS_CHAIN_FLTR]]
    return df_disp.sort_values("f1_score", ascending=False)

def evaluate_training_performance(essays, filter_to_predicted_tags=False, nearest_ref_only=False):
    proc_essays = get_coref_processed_essays(
                            essays=essays, format_ana_tags=True, 
                            ner_ch_filter=None, look_back_only=True,
                            filter_to_predicted_tags=filter_to_predicted_tags, 
                            max_ana_phrase_len=None, max_cref_phrase_len=None, 
                            pos_ana_filter=None, pos_ch_filter=None, 
                            nearest_ref_only=nearest_ref_only)
                        
    metrics = get_metrics_raw(proc_essays, expected_tags=all_ana_tags,  micro_only=True)
    row = metrics["MICRO_F1"]

    df_results = pd.DataFrame([row])
    return df_results


def blank_if_none(val):
    return "-" if (val is None or not val or str(val).lower() == "none" or str(val) == EMPTY) else val

def visualize_essay(e):
    for sent_ix in range(len(e.sentences)):       
        sent = e.sentences[sent_ix]

        # SENTENCE LEVEL TAGS / PREDICTIONS
        ana_tags = e.ana_tagged_sentences[sent_ix]
        coref_ids = e.pred_corefids[sent_ix]
        # ner_tags = e.pred_ner_tags_sentences[sent_ix]
        pos_tags = e.pred_pos_tags_sentences[sent_ix]
        ptags = e.pred_tagged_sentences[sent_ix]
        
        print("*" * 100)
        print("SENTENCE {sent_ix}".format(sent_ix=sent_ix))
        print("*" * 100)
        for wd_ix in range(len(sent)):
            pos_tag = pos_tags[wd_ix]  # POS tag

            word, act_tags = sent[wd_ix]  # ignore actual tags
            pred_tags = ptags[wd_ix]  # predict cc tag
            if type(pred_tags) == str:
                # force to be a set
                pred_tags = {pred_tags}

            is_ana_tag = ana_tags[wd_ix] == ANAPHORA
            wd_coref_ids = coref_ids[wd_ix]  # Set[str]
            
            # for display
            disp_act_tags = blank_if_none(",".join(act_tags.intersection(reg_tags)))
            
            disp_ana_tags = blank_if_none(",".join(act_tags.intersection(all_ana_tags)))
            disp_ana_tags = disp_ana_tags.replace("Anaphor:[","").replace("]","")
            if disp_ana_tags != "-":
                disp_ana_tags = "Ana:" + disp_ana_tags
            
            disp_pred_tags = blank_if_none(",".join(pred_tags.intersection(reg_tags)))
            
            disp_pred_ana_tags = blank_if_none(",".join(pred_tags.intersection(all_ana_tags)))
            disp_pred_ana_tags = disp_pred_ana_tags.replace("Anaphor:[","").replace("]","")
            if disp_pred_ana_tags != "-":
                disp_pred_ana_tags = "Ana:" + disp_pred_ana_tags
                            
            print("{wd_ix} {word}\t{act_tags}\t{ana_tags}\t| {pred_tags}\t{pred_ana_tags}\t{pos_tag}\t{is_ana_tag}\t{corefids}".format(
                wd_ix=str(wd_ix).ljust(2),
                word=word.ljust(20),
                act_tags=disp_act_tags.ljust(10),
                ana_tags=disp_ana_tags.ljust(10),
                pred_tags=disp_pred_tags,
                pred_ana_tags=disp_pred_ana_tags,
                pos_tag=blank_if_none(pos_tag),
                is_ana_tag= "X" if is_ana_tag else "-",
                corefids = blank_if_none(",".join(sorted(wd_coref_ids)))
            ))

essay = matching_essays[13]
tagged_essay = get_coref_processed_essays([essay], format_ana_tags=True, filter_to_predicted_tags=False)[0]
visualize_essay(tagged_essay)

