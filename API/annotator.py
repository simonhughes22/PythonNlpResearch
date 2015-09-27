# coding=utf-8
import Settings
from model_store import ModelStore
from window_based_tagger_config import get_config
from processessays import process_essays, build_spelling_corrector
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from BrattEssay import Essay, load_bratt_essays
import numpy as np

from featureextractortransformer import FeatureExtractorTransformer
from sent_feats_for_stacking import CAUSAL_REL, RESULT_REL, CAUSE_RESULT, get_sent_feature_for_stacking_from_tagging_model

from featureextractionfunctions import fact_extract_positional_word_features_stemmed, fact_extract_ngram_features_stemmed
from wordtagginghelper import flatten_to_wordlevel_feat_tags, test_classifier_per_code

from traceback import format_exc
from config import Config

import logging

def onlyascii(s):
    out = ""
    for char in s:
        if ord(char) > 127:
            out += ""
        else:
            out += char
    return out

def friendly_tag(tag):
    return tag.replace("Causer:", "").replace("Result:", "")

def cr_sort_key(cr):
    cr = cr.replace("5b", "5.5")
    # _'s last
    if cr[0] == "_":
        return (99999999, cr, cr, cr)
    # Casual second to last, ordered by the order of the cause then the effect
    if "->" in cr:
        cr = friendly_tag(cr)
        a,b = cr.split("->")
        if a.isdigit():
            a = float(a)
        if b.isdigit():
            b = float(b)
        return (9000, a,b, cr)
    # order causer's before results
    elif "Result:" in cr:
        cr = friendly_tag(cr)
        return (-1, float(cr),-1,cr)
    elif "Causer:" in cr:
        cr = friendly_tag(cr)
        return (-2, float(cr),-1,cr)
    else:
        #place regular tags first, numbers ahead of words
        if cr[0].isdigit():
            return (-10, float(cr),-1,cr)
        else:
            return (-10, 9999.9   ,-1,cr.lower())
    return (float(cr.split("->")[0]), cr) if cr.split("->")[0][0].isdigit() else (99999, cr)

class TaggedSentence(object):
    def __init__(self, sentence, codes, causal):
        self.tagged_words = []
        self.causal = causal
        self.codes = codes
        self.sentence = sentence

    def add_word_tags(self, tagged_words):
        self.tagged_words = tagged_words
        return self

class TaggedWord(object):
    def __init__(self, word, corrected_word, codes, causal):
        self.corrected_word = corrected_word
        self.word = word
        self.codes = codes
        self.causal = causal

class Annotator(object):

    @classmethod
    def from_config(cls, config_file):
        cfg = Config(config_file)
        return Annotator(cfg.models_folder, cfg.essays_folder)

    def __init__(self, models_folder, essays_folder):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        if not models_folder.endswith("/"):
            models_folder += "/"
        if not essays_folder.endswith("/"):
            essays_folder += "/"

        self.logger = logging.getLogger()
        cfg = get_config(essays_folder)
        self.config = cfg
        self.essays_folder = essays_folder

        # Create spell checker
        # Need annotations here purely to load the tags
        tagged_essays = load_bratt_essays(essays_folder, include_vague=cfg["include_vague"], include_normal=cfg["include_normal"], load_annotations=True)
        self.__set_tags_(tagged_essays)
        self.wd_sent_freq = defaultdict(int)
        self.spelling_corrector = build_spelling_corrector(tagged_essays, self.config["lower_case"], self.wd_sent_freq)

        offset = (self.config["window_size"] - 1) / 2

        unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
        biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

        extractors = [unigram_window_stemmed, biigram_window_stemmed]

        # most params below exist ONLY for the purposes of the hashing to and from disk
        self.feature_extractor = FeatureExtractorTransformer(extractors)

        # load models
        self.logger.info("Loading pickled models")
        store = ModelStore(models_folder=models_folder)

        self.feature_transformer =  store.get_transformer()
        self.logger.info("Loaded Transformer")
        self.tag_2_wd_classifier = store.get_tag_2_wd_classifier()
        self.logger.info("Loaded word tagging model")
        self.tag_2_sent_classifier = store.get_tag_2_sent_classifier()
        self.logger.info("Loaded sentence classifier")

    def annotate(self, essay_text):

        try:
            sentences = sent_tokenize(essay_text.strip())
            contents = "\n".join(sentences)

            essay = Essay(full_path=None, include_vague=self.config["include_vague"],
                          include_normal=self.config["include_normal"], load_annotations=False, essay_text=contents)

            processed_essays = process_essays(essays=[essay],
                                              spelling_corrector=self.spelling_corrector,
                                              wd_sent_freq=self.wd_sent_freq,
                                              remove_infrequent=self.config["remove_infrequent"],
                                              spelling_correct=self.config["spelling_correct"],
                                              replace_nums=self.config["replace_nums"],
                                              stem=self.config["stem"],
                                              remove_stop_words=self.config["remove_stop_words"],
                                              remove_punctuation=self.config["remove_punctuation"],
                                              lower_case=self.config["lower_case"])

            self.logger.info("Essay loaded successfully")
            essays_TD = self.feature_extractor.transform(processed_essays)

            wd_feats, _ = flatten_to_wordlevel_feat_tags(essays_TD)
            xs = self.feature_transformer.transform(wd_feats)

            wd_predictions_by_code = test_classifier_per_code(xs, self.tag_2_wd_classifier, self.wd_test_tags)

            dummy_wd_td_ys_bytag = defaultdict(lambda: np.asarray([0.0] * xs.shape[0]))
            sent_xs, sent_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(self.sent_input_feat_tags,
                                                                                             self.sent_input_interaction_tags,
                                                                                             essays_TD, xs,
                                                                                             dummy_wd_td_ys_bytag,
                                                                                             self.tag_2_wd_classifier,
                                                                                             sparse=True,
                                                                                             look_back=0)

            """ Test Stack Classifier """

            sent_predictions_by_code = test_classifier_per_code(sent_xs, self.tag_2_sent_classifier, self.sent_output_train_test_tags)

            """ Generate Return Values """
            essay_tags = self.__get_essay_tags_(sent_predictions_by_code)

            essay_type = None
            if "coral" in self.essays_folder.lower():
                essay_type = "CB"
            elif "skin" in self.essays_folder.lower():
                essay_type = "SC"
            else:
                raise Exception("Unknown essay type")

            raw_essay_tags = ",".join(sorted(essay_tags, key=cr_sort_key))

            t_words = self.__get_tagged_words_(essay, essays_TD[0], wd_predictions_by_code)
            t_sentences = self.__get_tagged_sentences_(essay, sent_predictions_by_code)

            tagged_sentences = [t_sent.add_word_tags(map(lambda twd: twd.__dict__, t_wds)).__dict__
                                for t_sent, t_wds in zip(t_sentences, t_words)]

            essay_codes, essay_causal = self.__format_essay_tags_(essay_tags)
            return {"tagged_sentences"  :   tagged_sentences,

                    "essay_codes"       :   essay_codes,
                    "essay_causal"      :   essay_causal,
                    "essay_category"    :   self.essay_category(raw_essay_tags, essay_type),

                    "raw_essay_tags"    :   raw_essay_tags
            }
        except Exception as x:
            self.logger.exception("An exception occured while annotating essay")
            return {"error": format_exc()}
        pass

    def __set_tags_(self, tagged_essays):

        MIN_TAG_FREQ = 5

        tag_freq = defaultdict(int)
        for essay in tagged_essays:
            for sentence in essay.tagged_sentences:
                un_tags = set()
                for word, tags in sentence:
                    for tag in tags:
                        if "5b" in tag:
                            continue
                        if      (tag[-1].isdigit() or tag in {"Causer", "explicit", "Result"} \
                                    or tag.startswith("Causer") or tag.startswith("Result") \
                                    or tag.startswith("explicit") or "->" in tag) \
                                and not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag):
                            # if not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag):
                            un_tags.add(tag)
                for tag in un_tags:
                    tag_freq[tag] += 1

        all_tags = list(tag_freq.keys())
        freq_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= MIN_TAG_FREQ)))
        non_causal = [t for t in freq_tags if "->" not in t]
        only_causal = [t for t in freq_tags if "->" in t]

        CAUSE_TAGS = ["Causer", "Result", "explicit"]
        CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]  # + ["explicit"]

        """ works best with all the pair-wise causal relation codes """
        # Include all tags for the output
        self.wd_test_tags = list(set(all_tags + CAUSE_TAGS))

        # tags from tagging model used to train the stacked model
        self.sent_input_feat_tags = list(set(freq_tags + CAUSE_TAGS))
        # find interactions between these predicted tags from the word tagger to feed to the sentence tagger
        self.sent_input_interaction_tags = list(set(non_causal + CAUSE_TAGS))
        # tags to train (as output) for the sentence based classifier
        self.sent_output_train_test_tags = list(set(all_tags + CAUSE_TAGS + CAUSAL_REL_TAGS))

    def essay_category(self, s, essay_type):

        essay_type = essay_type.strip().upper()

        if not s or s == "" or s == "nan":
            return 1
        splt = s.strip().split(",")
        splt = filter(lambda s: len(s.strip()) > 0, splt)
        regular = [t.strip() for t in splt if t[0].isdigit()]
        any_causal = [t.strip() for t in splt if "->" in t and (("Causer" in t and "Result" in t) or "C->R" in t)]
        causal = [t.strip() for t in splt if "->" in t and "Causer" in t and "Result" in t]
        if len(regular) == 0 and len(any_causal) == 0:
            return 1
        if len(any_causal) == 0:  # i.e. by this point regular must have some
            return 2  # no causal
        # if only one causal then must be 3
        elif len(any_causal) == 1 or len(causal) == 1:
            return 3
        # Map to Num->Num, e.g. Causer:3->Results:50 becomes 3->5
        # Also map 6 to 16 and 7 to 17 to enforce the relative size relationship

        def map_cb(code):
            return code.replace("6", "16").replace("7", "17")

        def map_sc(code):
            return code.replace("4", "14").replace("5", "15").replace("6", "16").replace("150", "50")

        is_cb = False
        is_sc = False
        if essay_type == "CB":
            is_cb = True
            crels = sorted(map(lambda t: map_cb(t.replace("Causer:", "").replace("Result:", "")).strip(), causal),
                           key=cr_sort_key)
        elif essay_type == "SC":
            is_sc = True
            crels = sorted(map(lambda t: map_sc(t.replace("Causer:", "").replace("Result:", "")).strip(), \
                               causal),
                           key=cr_sort_key)
        else:
            raise Exception("Unrecognized filename")

        un_results = set()
        # For each unique pairwise combination
        for a in crels:
            for b in crels:
                if cr_sort_key(b) >= cr_sort_key(a):  # don't compare each pair twice (a,b) == (b,a)
                    break
                # b is always the smaller of the two
                bc, br = b.split("->")
                ac, ar = a.split("->")
                # if result from a is causer for b
                if br.strip() == ac.strip():
                    un_results.add((b, a))

        if len(un_results) >= 1:
            #CB and 6->7->50 ONLY
            if len(un_results) == 1 and is_cb and ("16->17", "17->50") in un_results:
                return 4
            if len(un_results) <= 2 and is_sc:
                #4->5->6->50
                codes = set("14,15,16,50".split(","))
                un_results_cp = set(un_results)
                for a, b in un_results:
                    alhs, arhs = a.split("->")
                    blhs, brhs = b.split("->")
                    if alhs in codes and arhs in codes and blhs in codes and brhs in codes:
                        un_results_cp.remove((a, b))
                if len(un_results_cp) == 0:
                    return 4
            return 5
        else:
            return 3

    def __is_tag_to_return_(self, tag):
        return tag[0].isdigit() or ("->" in tag and "Causer" in tag)

    def __get_regular_tags_(self, pred_tags):
        r_tags = sorted(filter(lambda t: t[0].isdigit() and "->" not in t, pred_tags),
                        key=lambda s: (int(s), s) if s.isdigit() else ((-1, s)))
        str_r_tags = ",".join(r_tags)
        return str_r_tags

    def __get_causal_tags_(self, pred_tags):
        c_tags = sorted(filter(lambda t: "->" in t, pred_tags), key=cr_sort_key)
        str_c_tags = ",".join(c_tags)
        return str_c_tags

    def __get_tagged_sentences_(self, essay, sent_predictions_by_code):
        tagged_sents = []
        for i, sent in enumerate(essay.tagged_sentences):
            wds, _ = zip(*sent)
            str_sent = " ".join(wds)
            pred_tags = set()
            for tag, array in sent_predictions_by_code.items():
                if self.__is_tag_to_return_(tag):
                    if np.max(array[i]) == 1:
                        pred_tags.add(friendly_tag(tag))

            str_r_tags = self.__get_regular_tags_(pred_tags)
            str_c_tags = self.__get_causal_tags_(pred_tags)

            tagged_sents.append(TaggedSentence(str_sent, str_r_tags, str_c_tags ))
        return tagged_sents

    def __get_essay_tags_(self, sent_predictions_by_code):
        tags = set()

        for tag, array in sent_predictions_by_code.items():
            if np.max(array) == 1:
                tags.add(tag)

        return tags

    def __format_essay_tags_(self, tags):

        tags = map(lambda s: friendly_tag(s), filter(lambda t: self.__is_tag_to_return_(t), tags))

        str_r_tags = self.__get_regular_tags_(tags)
        str_c_tags = self.__get_causal_tags_(tags)

        if not str_r_tags:
            return "", str_c_tags
        elif not str_c_tags:
            return str_r_tags, ""
        else:
            return str_r_tags, str_c_tags

    def __fuzzy_match_(self, original, feat_wd):
        original = original.lower().strip()
        feat_wd = feat_wd.lower().strip()
        if original == feat_wd:
            return True
        if original[:3] == feat_wd[:3]:
            return True
        a = set(original)
        b = set(feat_wd)
        jaccard = float(len(a.intersection(b))) / float(len(a.union(b)))
        return jaccard >= 0.5

    def __align_wd_tags_(self, orig, feats):
        """
        Once processed, there may be a different number of words than in the original sentence
        Try and recover the tags for the original words by aligning the two using simple heuristics
        """
        if len(orig) < len(feats):
            raise Exception("align_wd_tags() : Original sentence is longer!")

        o_wds, _ = zip(*orig)
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
                for j in range(i, i + diff + 1):
                    new_tagged_wds.append((o_wds[j], new_tags[new_tag_ix]))
                    next_orig_wd = o_wds[j]
                    if self.__fuzzy_match_(next_orig_wd, feat_wd):
                        found = True
                        tagged_wds.extend(new_tagged_wds)
                        feat_offset += len(new_tagged_wds) - 1
                        break
                if not found:
                    raise Exception("No matching word found for index:%i and processed word:%s" % (i, feat_wd))
        return tagged_wds

    def __get_tagged_words_(self, original_essay, essay_TD, wd_predictions_by_code):
        tagged_sents = []
        # should be a one to one correspondance between words in essays_TD[0] and predictions
        i = 0
        for sent_ix, sent in enumerate(essay_TD.sentences):
            tmp_tagged_wds = []
            for wix, (feat) in enumerate(sent):
                word = feat.word
                tags = set()
                for tag in wd_predictions_by_code.keys():
                    if wd_predictions_by_code[tag][i] > 0:
                        tags.add(tag)
                i += 1
                tmp_tagged_wds.append((word, tags))

            # Now allign the predicted tags with the original words
            wds, aligned_tags = zip(*self.__align_wd_tags_(original_essay.tagged_sentences[sent_ix], tmp_tagged_wds))
            # spelling correct (needs to be after alignment)

            fr_aligned_tags = map(lambda tags: set(map(friendly_tag, tags)), aligned_tags)
            tagged_words = zip(wds, fr_aligned_tags)
            tagged_sents.append(map(lambda (wd, tags): TaggedWord(wd, self.spelling_corrector.correct(wd), self.__get_regular_tags_(tags), self.__get_causal_tags_(tags)), tagged_words))
        return tagged_sents

if __name__ == "__main__":

    import os
    cwd = os.getcwd()

    settings = Settings.Settings()
    folder = settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"

    annotator = Annotator(models_folder= cwd +"/Models/CB/", temp_folder=cwd+"/temp/", essays_folder=folder)
    d_annotations = annotator.annotate("""
Corals are living animals in the ocean.
Corals live in one place and dont really move alot.
Some corals have white on them and that is called "coral bleaching."
Coral Bleaching means that the coral is unhealthy and is trusting into a white color.
Normal water tempatures that the coral live in are 70-80 degrees.
But some of the waters are too cool like 3 to 10 degrees F.
Corals are also affected by storms because corals rely on the amounts of salt in the waters.
So when it storms the water tempatures and levels of salt will be all mest up and bad for the coral.
The storms have to be very extreme to make corals sick or unhealthy.
In the water if the tempature increases the amounts of dioxide will drop and willmake the coral unhealthy.
The water tempatures coral usally build their reefs in are 70-85 degrees F.
So those are the tempature range to keep them healthy.
Corals and zooanthellae algae have a relatioship together.
Most zooanthellae can not live without outside the corals bodies.
It is because there isnt enough nutrience to have the ocean do photosynthesis.
The zooanthellae rely on the coral to stay healthy, but the coral can get physical damage.
Coral bleaching is a physical damage to the corals.
Coral bleaching is also an example how the envionmental stressors can affect the relationships between the coral and the algae. //
    """)

    for sent, r_tags, c_tags in d_annotations["tagged_sentences"]:
        print "\"" + sent + "\"", r_tags, c_tags
    print ""

    for sent in d_annotations["tagged_words"]:
        for wd, r_tags, c_tags in sent:
            print str((wd, r_tags, c_tags))
        print ""
    print ""
    print "Essay Tags"
    print d_annotations["essay_tags"]
    print "\nRaw tags"
    print d_annotations["raw_essay_tags"]
    print "\nEssay Category"
    print d_annotations["essay_category"]
