from GwExperimentBase import *
from OrderedDecisionTree import OrderedDecisionTree, not_a_followed_by_b, compute_attribute_values_forward_only
from SkipGramGenerator import skip_gram_matches


class Codes_ClassifyUsingOrderedDecisionTree(GwExperimentBase):

    def __init__(self):
        pass

    def create_classifier(self, code):
        def cls_create(xs, ys):

            dt = OrderedDecisionTree()
            dt.fn_attribute_val_extractor = compute_attribute_values_forward_only
            dt.fit(xs, ys)
            return dt
        return cls_create
    
    def get_vector_space(self, tokenized_docs):
        return (tokenized_docs, None)

    def get_training_data(self, tokenized_docs, id2word):
        return tokenized_docs

if __name__ == "__main__":

    criterion = 'entropy'
    cl = Codes_ClassifyUsingOrderedDecisionTree()
    cl.Run(Codes_ClassifyUsingOrderedDecisionTree.__name__ + ".txt", spelling_correct=False)