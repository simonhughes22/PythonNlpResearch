from sklearn.svm import SVC
from GwExperimentBase import *
from WindowSplitter import split_into_windows
from TermFrequency import TermFrequency

class Codes_ClassifyUsingSVM_MaxWindowed_Binary(GwExperimentBase):

    def __init__(self, C, window_size):
        self.C = C
        self.vectorizer = None
        self.window_size = window_size
        pass #no params     

    def create_classifier(self, code):

        def cls_create(xs, ys):

            new_xs, new_ys = [], []
            for x, y in zip(xs, ys):
                win_xs = self.get_binary_windows(x)
                new_xs.extend(win_xs)
                new_ys.extend([y for i in range(len(win_xs))])

            svc = SVC(C=self.C, probability=True)
            svc.fit(new_xs, new_ys)
            return svc
        return cls_create
    
    def classify(self):
        def classify(classifier, vd):

            predictions = []
            for new_xs in vd:
                windows = self.get_binary_windows(new_xs)
                probs = classifier.predict_proba(windows)
                # probs is a list of probabilites, sorted in order of the class (negative, then positive)
                max_negative_prob = max(map(lambda tpl: tpl[0], probs))
                max_positive_prob = max(map(lambda tpl: tpl[1], probs))

                if max_negative_prob >= max_positive_prob:
                    predictions.append(-1)
                else:
                    predictions.append( 1)

            return predictions
        return classify

    def get_binary_windows(self, x):
        new_xs = []
        windows = split_into_windows(x, self.window_size)
        for w in windows:
            binary_vector = self.tf.to_binary_vector(w)
            new_xs.append(binary_vector)
        return new_xs

    def get_vector_space(self, tokenized_docs):
        return (tokenized_docs, {})

    def get_training_data(self, tokenized_docs, id2word):
        self.tf = TermFrequency(tokenized_docs)
        return tokenized_docs

    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":

    C = 3.0
    window_size = 5
    cl = Codes_ClassifyUsingSVM_MaxWindowed_Binary(C, window_size)
    fname = Codes_ClassifyUsingSVM_MaxWindowed_Binary.__name__ + "_C_" + str(C) + "_win_" + str(window_size) + ".txt"

    (mean_metrics, wt_mean_metrics) = cl.Run(fname, one_fold=False)
