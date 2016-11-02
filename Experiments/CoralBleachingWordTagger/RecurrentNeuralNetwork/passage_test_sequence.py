from passage.layers import Embedding
from passage.layers import GatedRecurrent
from passage.layers import LstmRecurrent
from passage.layers import Dense

from passage.models import RNN
from passage.utils import save, load
from passage.preprocessing import Tokenizer
from passage.theano_utils import intX
from passage.iterators import SortedPadded
import theano.tensor as T
from IterableFP import flatten

#tokenizer = Tokenizer()
#SH: doesn't work for some reason
#train_tokens = tokenizer.fit_transform(["The big fat frog jumped out of the pond","frogs are amphibians", "toads are similar to frogs"])

train_tokens = [[1, 2, 4, 3, 6], [1, 2, 3], [3, 1, 2, 4, 3]]
num_feats = len(set(flatten(train_tokens)))

def get_labels(id):
    if id == 3:
        return [1, 0]
    else:
        return [0, 1]

seq_labels = map(lambda (l): map(get_labels ,  l), train_tokens)

layers = [
    Embedding(size=128, n_features=num_feats),
    GatedRecurrent(size=128, seq_output=True),
    Dense(size=num_feats, activation='softmax')
]

#iterator = SortedPadded(y_pad=True, y_dtype=intX)
#iterator = SortedPadded(y_dtype=intX)

#model = RNN(layers=layers, cost='seq_cce', iterator=iterator, Y=T.imatrix())
model = RNN(layers=layers, cost='seq_cce')
#model.fit(train_tokens, [1,0,1])
model.fit(train_tokens, train_tokens)

#model.predict(tokenizer.transform(["Frogs are awesome", "frogs are amphibious"]))
model.predict(train_tokens)
save(model, 'save_test.pkl')
model = load('save_test.pkl')


""" This model, although doing sequential prediction, predicts a tag per document not per word. """