# coding=utf-8
import Settings
from model_store import ModelStore
from window_based_tagger_config import get_config
from processessays import process_essays, build_spelling_corrector
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from BrattEssay import Essay, load_bratt_essays

from featureextractortransformer import FeatureExtractorTransformer
from sent_feats_for_stacking import *
from load_data import load_process_essays_without_annotations

from featureextractionfunctions import *
from wordtagginghelper import *

from traceback import format_exc

import logging

def onlyascii(s):
    out = ""
    for char in s:
        if ord(char) > 127:
            out += ""
        else:
            out += char
    return out

class Annotator(object):

    def __init__(self, models_folder, temp_folder, essays_folder):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        if not models_folder.endswith("/"):
            models_folder += "/"
        if not temp_folder.endswith("/"):
            temp_folder += "/"
        if not essays_folder.endswith("/"):
            essays_folder += "/"

        self.logger = logging.getLogger()
        self.temp_folder = temp_folder
        cfg = get_config(temp_folder)
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

    def annotate(self, essay_text):

        try:
            # expects a new line per sentence
            sentences = sent_tokenize(essay_text.strip())
            contents = "\n".join(sentences)

            fname = self.temp_folder + "essay.txt"
            with open(fname, 'w"') as f:
                f.write(contents)

            essay = Essay(fname, include_vague=self.config["include_vague"], include_normal=self.config["include_normal"], load_annotations=False)
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

            td_feats, _ = flatten_to_wordlevel_feat_tags(essays_TD)
            td_X = self.feature_transformer.transform(td_feats)
            td_wd_predictions_by_code = test_classifier_per_code(td_X, self.tag_2_wd_classifier, self.wd_test_tags)

            dummy_wd_td_ys_bytag = defaultdict(lambda: np.asarray([0.0] * td_X.shape[0]))
            sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(self.sent_input_feat_tags,
                                                                                             self.sent_input_interaction_tags,
                                                                                             essays_TD, td_X,
                                                                                             dummy_wd_td_ys_bytag,
                                                                                             self.tag_2_wd_classifier,
                                                                                             sparse=True,
                                                                                             look_back=0)

            """ Test Stack Classifier """
            td_sent_predictions_by_code \
                = test_classifier_per_code(sent_td_xs, self.tag_2_sent_classifier, self.sent_output_train_test_tags)

            return essay
        except Exception as x:
            self.logger.exception("An exception occured while annotating essay")
            return {"error": format_exc()}
        pass


if __name__ == "__main__":

    import os
    cwd = os.getcwd()

    settings = Settings.Settings()
    folder = settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"

    annotator = Annotator(models_folder= cwd +"/Models/CB/", temp_folder=cwd+"/temp/", essays_folder=folder)
    annotations = annotator.annotate("""
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
    for sent in annotations.tagged_sentences:
        print sent
    pass
