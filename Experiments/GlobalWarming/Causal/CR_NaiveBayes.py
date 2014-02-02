from GwCrExperimentBase import GwCrExperimentBase
from GwExperimentBase import *

class CR_NaiveBayes(GwCrExperimentBase):

    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)

    def create_classifier(self, code):
        def create_nb(xs, ys):
            td = zip(xs, ys)
            return nltk.NaiveBayesClassifier.train(td)
        return create_nb

    def classify(self):
        def classify(nb, vd):
            return [nb.classify(d) for d in vd]
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_binary_data(distance_matrix, id2word)

if __name__ == "__main__":

    cl = CR_NaiveBayes()
    cl.Run("CR_NaiveBayes.txt")