from GwExperimentBase import *
from OrderedDecisionTree import OrderedDecisionTree

class Codes_ClassifyUsingOrderedDecisionTree2(GwExperimentBase):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        pass

    def create_classifier(self, code):
        def cls_create(xs, ys):

            dt = OrderedDecisionTree(self.max_depth)
            dt.fit(xs, ys)
            return dt
        return cls_create
    
    def get_vector_space(self, tokenized_docs):
        return (tokenized_docs, None)

    def get_training_data(self, tokenized_docs, id2word):
        return tokenized_docs

if __name__ == "__main__":

    max_depth = 25
    cl = Codes_ClassifyUsingOrderedDecisionTree2(max_depth)

    cl.Run(Codes_ClassifyUsingOrderedDecisionTree2.__name__ + "_MaxDepth_" + str(max_depth) + "_.txt", spelling_correct=False)