from FindFiles import find_files
from collections import defaultdict
import dill

def get_essays(folder, partition):
    essay_files = find_files(folder, regex=".*.dill")
    if partition == "Training":
        essay_files = [e for e in essay_files if "train" in e]
    else:
        essay_files = [e for e in essay_files if "test" in e]
    assert len(essay_files) == 1
    print("Found file", essay_files[0])
    with open(essay_files[0], "rb") as f:
        loaded_essays = dill.load(f)
    return loaded_essays
    
def validate_essays(essays):
    for e in essays:    
        # map coref ids to sent_ix, wd_ix tuples
        # now look for ana tags that are also corefs, and cross reference
        for sent_ix in range(len(e.sentences)):
            sent     = e.sentences[sent_ix]
            ana_tags = e.ana_tagged_sentences[sent_ix]
            coref_ids= e.pred_corefids[sent_ix]
            ner_tags = e.pred_ner_tags_sentences[sent_ix]
            pos_tags = e.pred_pos_tags_sentences[sent_ix]
            ptags    = e.pred_tagged_sentences[sent_ix]

            assert len(sent) == len(coref_ids)

            assert len(sent) == len(ana_tags) == len(coref_ids) == len(ner_tags) == len(pos_tags) == len(ptags),\
                (len(sent), len(ana_tags), len(coref_ids), len(ner_tags), len(pos_tags), len(ptags), e.name, sent_ix)
            assert len(sent) > 0
    print("Essays validated")  
      
def tally_essay_attributes(essays, attribute_name="pred_pos_tags_sentences"):
    tally = defaultdict(int)
    for e in essays:
        nested_list = getattr(e, attribute_name)
        for lst in nested_list:
            for item in lst:
                if type(item) == str:
                    tally[item] +=1
                elif type(item) == set:
                    for i in item:
                        tally[i] +=1
                else:
                    raise Exception("Unexpected item type")
    return tally
