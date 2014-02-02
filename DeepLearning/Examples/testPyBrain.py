import pybrain
import pybrain.unsupervised
import pybrain.unsupervised.trainers
import pybrain.unsupervised.trainers.deepbelief
from pybrain.tools.shortcuts import buildNetwork
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import UnsupervisedDataSet
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer
from pybrain.structure.connections import FullConnection, IdentityConnection

def get_binary_data(xs):
    import WordTokenizer
    import TermFrequency
    import MatrixHelper
    import Converter
    
    tokenized_docs = WordTokenizer.tokenize(xs.documents, min_word_count=5)
    tf = TermFrequency.TermFrequency(tokenized_docs)

    arr = MatrixHelper.gensim_to_numpy_array(tf.distance_matrix, None, 0, Converter.to_binary)
    return arr

def custom_build_network(layer_sizes):
    net = FeedForwardNetwork()
    
    layers = []
    inp = SigmoidLayer(layer_sizes[0], name = 'visible')
    h1 = SigmoidLayer(layer_sizes[1], name = 'hidden1')
    h2 = SigmoidLayer(layer_sizes[2], name = 'hidden2')
    out = SigmoidLayer(layer_sizes[3], name = 'out')
    bias = BiasUnit(name = 'bias')
    
    net.addInputModule(inp)
    net.addModule(h1)
    net.addModule(h2)
    net.addOutputModule(out)
    net.addModule(bias)
    
    net.addConnection(FullConnection(inp, h1))
    net.addConnection(FullConnection(h1, h2))
    net.addConnection(FullConnection(h2, out))
    
    net.addConnection(FullConnection(bias, h1))
    net.addConnection(FullConnection(bias, h2))
    net.addConnection(FullConnection(bias, out))
    
    
    net.sortModules()
    return net
    
if __name__ == "__main__":
    
    import GwData
    data = GwData.GwData()
    xs = get_binary_data(data)
    ys = data.labels_for("50")
    
    sdataset = SupervisedDataSet(xs.shape[1], 1)
    udataset = UnsupervisedDataSet(xs.shape[1])
    for i,x in enumerate(xs):
        sdataset.addSample(x, ys[i])
        udataset.addSample(x)
    
    epochs = 100
    layerDims = [xs.shape[1], 300, 100, 2]    
    
    #net = buildNetwork(*layerDims)
    net = custom_build_network(layerDims)

    trainer = DeepBeliefTrainer(net, dataset=udataset)
    #trainer = DeepBeliefTrainer(net, dataset=sdataset)
    trainer.trainEpochs(epochs)