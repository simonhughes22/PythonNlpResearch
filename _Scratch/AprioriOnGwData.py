__author__ = 'simon.hughes'

import GwData
import WordTokenizer
from Apriori import apriori, print_rules

code = "50"

data = GwData.GwData()
xs = WordTokenizer.tokenize(data.documents, spelling_correct=False)
ys = ["CODE_50" if y == 1 else "NOT_50" for y in  data.labels_for(code)]

inputs = [ x + [y] for x,y in zip(xs, ys) ]

rules, support_data= apriori(inputs, min_support=0.025, max_k=5)
print_rules(rules, support_data, inputs, min_size=2)

#print len(inputs)