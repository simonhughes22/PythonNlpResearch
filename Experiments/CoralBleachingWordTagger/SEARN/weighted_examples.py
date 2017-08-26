from collections import defaultdict
from typing import List, Optional, Any

class WeightedExamples(object):
    def __init__(self, labels=None, positive_value=1):
        """
        In the multiclass case (unweighted), pass in labels=None

        @param labels: Optional[List[str]]
        @param positive_value: int
        """
        self.xs = []
        self.classes = labels
        self.positive_value = positive_value
        self.labels  = defaultdict(list)  # list of ints
        self.weights = defaultdict(list)  # list of floats
        self.all_labels = []

    def add(self, x, actual_lbl, weights=None):
        self.xs.append(x)
        self.all_labels.append(actual_lbl)

        if self.classes:
            for lbl in self.classes:
                val = self.positive_value if lbl == actual_lbl else -1
                self.labels[lbl].append(val)
                weight = 1
                if weights:
                    weight = weights[lbl]
                self.weights[lbl].append(weight)

    def get_labels(self):
        return self.all_labels

    def get_labels_for(self, lbl)->List[Any]:
        if not self.classes:
            raise Exception("Labels not supported, multi-class use case")
        return self.labels[lbl]

    def get_weights_for(self, lbl: str)->List[float]:
        if not self.classes:
            raise Exception("Weights not supported, multi-class use case")
        return self.weights[lbl]
