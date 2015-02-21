from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(["The frog jumped out of the pond","frogs are amphibians", "toads are similar to frogs"])

layers = [
    Embedding(size=128, n_features=tokenizer.n_features),
    GatedRecurrent(size=128),
    Dense(size=1, activation='sigmoid')
]

model = RNN(layers=layers, cost='BinaryCrossEntropy')
model.fit(train_tokens, [1,0,1])

model.predict(tokenizer.transform(["Frogs are awesome", "frogs are amphibious"]))
save(model, 'save_test.pkl')
model = load('save_test.pkl')