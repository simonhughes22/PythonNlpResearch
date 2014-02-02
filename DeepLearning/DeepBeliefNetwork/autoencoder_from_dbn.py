import numpy as np
import deepnet
import backprop
import cPickle as pickle
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from deepnet import *

def demo_autoencoder():

    #load and norm the xs
    import GwData
    xs = GwData.GwData.as_binary()

    #OLD - MNIST training
    #xs = np.load('scaled_images.npy')
    #xs = np.asarray(xs, dtype='float32')
    #xs /= 255.0
    
    #set up and train the initial deepnet
    dnn = deepnet.DeepNet([xs.shape[1], 300, 150], ['sigmoid','sigmoid','sigmoid'])
    dnn.train(xs, [500, 500], [0.25, 0.1])
    #save the trained deepnet
    pickle.dump(dnn, file('pretrained.pkl','wb'))
    #unroll the deepnet into an autoencoder
    autoenc = unroll_network(dnn.network)
    ##fine-tune with backprop
    mlp = backprop.NeuralNet(network=autoenc)
    
    trained = mlp.train(mlp.network, xs, xs, max_iter=30, 
                        validErrFunc='reconstruction', 
                        targetCost='linSquaredErr')
    ##save
    pickle.dump(trained, file('network.pkl','wb'))

def demo_simple_autoencoder():

    #load and norm the xs
    import GwData
    xs = GwData.GwData.as_binary()
    
    mlp = backprop.NeuralNet(None, [xs.shape[1], 500, xs.shape[1]], ['sigmoid','sigmoid','sigmoid'])
    
    trained = mlp.train(mlp.network, xs, xs, max_iter=30, 
                        validErrFunc='reconstruction', 
                        targetCost='linSquaredErr', initialfit = 0, cg_iter = 5)
    ##save
    pickle.dump(trained, file('network.pkl','wb'))

def unroll_network(network):
    '''
    Takes a pre-trained network and treats it as an encoder network. The decoder
    network is constructed by inverting the encoder. The decoder is then appended
    to the input network to produce an autoencoder.
    '''
    decoder = []
    encoder = []
    for i in range(len(network)):
        elayer = backprop.Layer(network[i].W.T, network[i].hbias, network[i].n_hidden, network[i].hidtype)
        dlayer = backprop.Layer(network[i].W, network[i].vbias, network[i].n_visible, network[i].vistype)
        encoder.append(elayer)
        decoder.append(dlayer)
    decoder.reverse()
    encoder.extend(decoder)
    return encoder

def save_net_as_mat(pickled_net):
    '''
    Takes the network pickle file saved in demo_autoencoder and saves it as a .mat
    file for use with matlab
    '''
    network = pickle.load(file(pickled_net,'rb'))
    mdic = {}
    for i in range(len(network)/2):
        mdic['W%d'%(i+1)] = network[i].W.as_numpy_array()
        mdic['b%d'%(i+1)] = network[i].hbias.as_numpy_array()
        mdic['hidtype%d'%(i+1)] = network[i].hidtype
    scipy.io.savemat('network.mat', mdic)

def get_data(xs):
    import WordTokenizer
    import TermFrequency
    import MatrixHelper
    import Converter
    
    tokenized_docs = WordTokenizer.tokenize(xs.documents, min_word_count=5)
    tf = TermFrequency.TermFrequency(tokenized_docs)

    arr = MatrixHelper.gensim_to_numpy_array(tf.distance_matrix, None, 0, Converter.to_binary)
    return arr

def to_feed_forward_network(dbn, top_layers):
    network = dbn.network
    '''
    Takes a pre-trained network and treats it as an top_layers network. The decoder
    network is constructed by inverting the top_layers. The decoder is then appended
    to the input network to produce an autoencoder.
    '''
    import backprop
    layers = []
    for i in range(len(network)):
        layer = backprop.Layer(network[i].W.T, network[i].hbias, network[i].n_hidden, network[i].hidtype)
        layers.append(layer)

    net = layers + top_layers
    mlp = backprop.NeuralNet(network=net)
    return mlp

def run_supervised():
    import GwData
    gwData = GwData.GwData()
    
    xs = get_data(gwData)
    
    def flip(i):
        if i == 0:
            return 1
        return 0

    ys = [[lbl, flip(lbl)] for lbl in gwData.labels_for("50")]

    xs = np.array(xs)
    ys = np.array(ys)

    td_size = 2500

    td_x = xs[0:td_size]
    vd_x = xs[td_size:]
    
    dbnetwork = DeepNet([td_x.shape[1], 600, 400], ['sigmoid', 'sigmoid', 'sigmoid'])
    dbnetwork.train(td_x, [1000, 1000], [0.1, 0.1])
    out = dbnetwork.run_through_network(xs)
  
    top_layer = backprop.NeuralNet(layer_sizes = [out.shape[1], int(out.shape[1]), 2], layer_types = ['sigmoid', 'sigmoid', 'sigmoid'])
    
    o_td_x = out[0:td_size]
    o_vd_x = out[td_size:]

    td_y = ys[0:td_size]
    vd_y = ys[td_size:]
    
    top_layers = top_layer.train(top_layer.network, o_td_x, td_y, o_vd_x, vd_y, 10, 'classification', 'crossEntropy', 0, 25)
  
    #TODO We need to train a top layer neural network from the top DBNN layer to the output
    #TODO Then we create a final network composed of the two concatenated together
    mlp = to_feed_forward_network(dbnetwork, top_layers)
    trained = mlp.train(mlp.network, td_x, td_y, vd_x, vd_y, max_iter=30, validErrFunc='classification', targetCost='crossEntropy')
    
    print out.shape
    np.save('output.npy', out)


def visualize_results(netfile, datafile):
    network = pickle.load(file(netfile, 'rb'))
    #network = unroll_network(dnn.network)
    xs = np.load(datafile)
    xs = np.asarray(xs, dtype='float32')
    xs /= 255.0
    mlp = backprop.NeuralNet(network=network)
    recon = mlp.run_up_through_network(xs, network)
    inds = np.arange(recon.shape[0])
    np.random.shuffle(inds)
    for i in range(10):
        dim = int(np.sqrt(xs.shape[1]))
        orig = xs[inds[i]].reshape((dim,dim))
        rec = recon[inds[i]].reshape((dim,dim))
        plt.figure(i)
        ax = plt.subplot(211)
        #plt.imshow(orig, cmap=cm.gray)
        ax.set_yticks([])
        ax.set_xticks([])
        ax = plt.subplot(212)
        #plt.imshow(rec, cmap=cm.gray)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.savefig('img_%d.jpg'%(inds[i]))

if __name__ == "__main__":
    #demo_autoencoder()
    demo_simple_autoencoder()
    #visualize_results('network.pkl','scaled_images.npy')