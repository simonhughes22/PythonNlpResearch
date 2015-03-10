__author__ = 'simon.hughes'

import os
import cPickle as pickle

class ModelStore():
    def __init__(self):
        models_folder = os.getcwd() + "/PickledModels/"
        print "Loading models from", models_folder
        self.models_folder = models_folder
        self.feat_transform_file = models_folder + "feat_extractor_pickled.p"
        self.tag_2_wd_classifiers_file = models_folder + "tag_2_wd_classifier_pickled.p"
        self.tag_2_sent_classifiers_file = models_folder + "tag_2_sent_classifier_pickled.p"

    def __store_model_(self, obj, fname):
        with open(fname, "w+") as f:
            pickle.dump(obj, f)

    def store(self, feat_transform, tag_2_wd_classifier, tag_2_sent_classifier):
        self.__store_model_(feat_transform,         self.feat_transform_file)
        self.__store_model_(tag_2_wd_classifier,    self.tag_2_wd_classifiers_file)
        self.__store_model_(tag_2_sent_classifier,  self.tag_2_sent_classifiers_file)

    def __load_model_(self, fname):
        with open(fname, "r+") as f:
            return pickle.load(f)

    def get_transformer(self):
        return self.__load_model_(self.feat_transform_file)

    def get_tag_2_wd_classifier(self):
        return self.__load_model_(self.tag_2_wd_classifiers_file)

    def get_tag_2_sent_classifier(self):
        return self.__load_model_(self.tag_2_sent_classifiers_file)
