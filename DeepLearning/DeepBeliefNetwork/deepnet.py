import sys
import os
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'gnumpy'))
import gnumpy as gp
import numpy as np
import backprop

class RBM(object):
    ''' 
    This class implements a restricted Bolzmann machine using gnumpy,
    which runs on a gpu if cudamat is installed
    
    args:
        int n_visible:    the number of visible units
        int n_hidden:     the number of hidden units, default is n_visible
        string vistype:   type of units for visible layer, default 'sigmoid'
        string hidtype:   type of units for hidden layer, default 'sigmoid'
        array W:          the 2d weight matrix, default None
        array hbias:      the bias weights for the hidden layer, default None
        array vbias:      the bias weights for the visible layer, default None
        int batch_size:   default 128

        if W, hbias, vbias are left as None (default), they will be created and 
            initialized automatically.

    methods:
        train(int num_epochs, array hidden, bool sample)
        prop_up(array xs)
        prop_down(array xs)
        hidden_state(array xs)

    variables:
        array wu_vh:  the weight update array which can be reused
        array wu_v:   the update array for vbias
        array wu_h:   the update array for hbias

    '''

    def __init__(self, n_visible, n_hidden=None, vistype='sigmoid', 
            hidtype='sigmoid', W=None, hbias=None, vbias=None, batch_size=100):
        # initialize parameters
        self.SIZE_LIMIT = 80000000 # the size of the largest gpu array
        self.vistype = vistype
        self.hidtype = hidtype
        self.batch_size = batch_size
        self.n_visible = n_visible
        if n_hidden is None:
            n_hidden = self.n_visible
        self.n_hidden = n_hidden
        n = self.n_visible*self.n_hidden + self.n_hidden
        bound = 2.38 / np.sqrt(n)
        if W is None:
            W = np.zeros((self.n_visible, self.n_hidden))
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    W[i,j] = np.random.uniform(-bound, bound)
        W = gp.garray(W)
        self.W = W
        if vbias is None:
            vbias = gp.zeros(self.n_visible)
        else:
            vbias = gp.garray(vbias)
        self.vbias = vbias
        if hbias is None:
            hbias = np.zeros((self.n_hidden,))
            for i in range(self.n_hidden):
                hbias[i] = np.random.uniform(-bound, bound)
        hbias = gp.garray(hbias)
        self.hbias = hbias
        #initialize updates
        self.wu_vh = gp.zeros((self.n_visible, self.n_hidden))
        self.wu_v = gp.zeros(self.n_visible)
        self.wu_h = gp.zeros(self.n_hidden)
        
        self.np_array_type = type(np.array([]))

    def train(self, fulldata, num_epochs, eta=0.01, hidden=None, sample=False, early_stop=True, verbose = True):
        ''' 
        Method to learn the weights of the RBM.

        args: 
            array fulldata: the training xs
            int num_epochs: the number of times to run through the training xs
            float eta:      the learning rate, default 0.01
            array hidden:   optional array specifying the hidden representation
                            to learn (for use in a translational-RBM)
            bool sample:    specifies whether training should use sampling, 
                            default False
            bool early_stop: whether to use early stopping, default True

        '''
        if len(fulldata) == 0:
            return
        
        if type(fulldata) != self.np_array_type  or type(fulldata[0]) != self.np_array_type:
            fulldata = np.array([np.array(r) for r in fulldata])
        
        if hidden is not None:
            # check that there is a hidden rep for each xs row
            assert hidden.shape[0] == xs.shape[0]
            # check that we have the right number of hidden units
            assert hidden.shape[1] == self.n_hidden

        # these parameters control momentum changes
        initial_momentum = 0.5
        final_momentum = 0.9
        momentum_iter = 5

        # when dealing with large arrays, we have to break the xs into
        # manageable chunks to avoid out of memory mae
        num_rows = fulldata.shape[0]
        
        err_hist = [] # keep track of the errors for early stopping
        
        for epoch in range(num_epochs):
            if epoch <= momentum_iter:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            
            mae = []
            if verbose:
                print "Training epoch %d of %d," %(epoch+1, num_epochs),
            
            num_batches = num_rows/self.batch_size + 1
                    
            xs = gp.garray(fulldata)
            if hidden is not None:
                hid_chunk = gp.garray(hidden)

            for batch in range(num_batches):
                # positive phase
                if num_batches == 1:
                    v1 = xs
                else:
                    v1 = xs[batch*self.batch_size:(batch+1)*self.batch_size]
                
                if len(v1) == 0:
                    continue
                
                if hidden is None:
                    h1 = self.prop_up(v1)
                else:
                    if num_batches == 1:
                        h1 = hid_chunk
                    else:
                        h1 = hid_chunk[batch*self.batch_size:(batch+1)*self.batch_size]

                # negative phase
                if sample:
                    hSampled = h1.rand() < h1
                    v2 = self.prop_down(hSampled)
                else:
                    v2 = self.prop_down(h1)
                h2 = self.prop_up(v2)
                
                # update weights
                self.wu_vh = self.wu_vh * momentum + gp.dot(v1.T, h1) - gp.dot(v2.T, h2)
                self.wu_v = self.wu_v * momentum + v1.sum(0) - v2.sum(0)
                self.wu_h = self.wu_h * momentum + h1.sum(0) - h2.sum(0)

                self.W += self.wu_vh * (eta/self.batch_size)
                self.vbias += self.wu_v * (eta/self.batch_size)
                self.hbias += self.wu_h * (eta/self.batch_size)
                
                # calculate reconstruction error
                error = gp.abs(v2 - v1)
                
                #mae.append(error.euclid_norm()**2/(self.n_visible*self.batch_size))
                mae.append(gp.mean(error))
              
            err_hist.append(np.mean(mae))
            if verbose:
                print " mean absolute error: "+ str(np.mean(mae))
                
            # early stopping
            if early_stop:
                recent_err = np.mean(err_hist[epoch-50:epoch])
                early_err = np.mean(err_hist[epoch-200:epoch-150])
                if (epoch > 250) and ((recent_err * 1.2) > early_err):
                    break

    def prop_up(self, xs):
        '''
        Method to return the hidden representation given xs on the visible layer.

        args:
            array xs:         the xs on the visible layer
        returns:
            array hid:   the probabilisitic activation of the hidden layer
        
        '''
        hid = gp.dot(xs, self.W) + self.hbias
        if self.hidtype == 'sigmoid':
            return hid.logistic()
        else:
            return hid

    def prop_down(self, xs):
        '''
        Method to return the visible representation given the hidden

        args:
            array xs:         the hidden representation
        returns:
            array vis:   the activation of the visible layer
        '''
        vis = gp.dot(xs, self.W.T) + self.vbias
        if self.vistype == 'sigmoid':
            return vis.logistic()
        else:
            return vis

    def hidden_state(self, xs):
        '''
        Method to sample from the hidden representation given the visible

        args:
            array xs:  the xs on the visible layer
        returns:
            array hSampled: the binary representation of the hidden layer activation
        '''
        hid = self.prop_up(xs)
        hSampled = hid.rand() < hid
        return hSampled

class Holder(object):
    '''
    Objects of this class hold values of the RBMs in numpy arrays to free up space 
    on the GPU
    '''
    def __init__(self, rbm):
        self.W = rbm.W.as_numpy_array()
        self.hbias = rbm.hbias.as_numpy_array()
        self.vbias = rbm.vbias.as_numpy_array()
        self.n_hidden = rbm.n_hidden
        self.n_visible = rbm.n_visible
        self.hidtype = rbm.hidtype
        self.vistype = rbm.vistype

    def prop_up(self, xs):
        hid = np.dot(xs, self.W) + self.hbias
        if self.hidtype == 'sigmoid':
            return 1./(1. + np.exp(-hid)) 
        else:
            return hid

class DeepNet(object):
    '''
    A class to implement a deep neural network

    args:
        list[int] layer_sizes: defines the number and size of top_layers 
        list[str] layer_types: defines layer types, 'sigmoid' or 'gaussian'

    methods: 
        train
        run_through_network
    '''
    def __init__(self, layer_sizes, layer_types, sample = False):
        assert len(layer_sizes) == len(layer_types)
        self.layer_sizes = layer_sizes
        self.layer_types = layer_types
        self.sample = sample
        
    def train(self, xs, epochs, eta, early_stop = True):
        '''
        Trains the deep net one RBM at a time

        args:
            array xs:         the training xs (a gnumpy.array)
            list[int] epochs:   the number of training epochs for each RBM
            float eta:          the learning rate
        '''
        top_layers = []
        vis = xs
        for i in range(len(self.layer_sizes)-1):
            print "Pretraining RBM %d, vis=%d, hid=%d" % (i+1, self.layer_sizes[i],
                    self.layer_sizes[i+1])
            g_rbm = RBM(self.layer_sizes[i], self.layer_sizes[i+1], self.layer_types[i], self.layer_types[i+1])
            g_rbm.train(vis, epochs[i], eta[i], sample = self.sample, early_stop = early_stop)
            hid = self.get_activation(g_rbm, vis)
            vis = hid
            n_rbm = Holder(g_rbm)
            top_layers.append(n_rbm)
            gp.free_reuse_cache()
        self.network = top_layers
        return vis
    
    def get_activation(self, rbm, xs):
        #hid = np.zeros((xs.shape[0], dbn.n_hidden))
        hid = (rbm.prop_up(xs)).as_numpy_array()
        return hid

    def get_visible(self, rbm, xs):
        hid = (rbm.prop_down(xs)).as_numpy_array()
        return hid

    def run_through_network(self, xs):
        hid = xs
        for n_rbm in self.network:
            vis = gp.garray(hid)
            g_rbm = RBM(n_rbm.n_visible, n_rbm.n_hidden, n_rbm.vistype, 
                    n_rbm.hidtype, n_rbm.W, n_rbm.hbias, n_rbm.vbias)
            hid = self.get_activation(g_rbm, hid)
            gp.free_reuse_cache()
        return hid

    def run_down_through_network(self, xs):
        hid = xs
        
        copy = self.network[:]
        copy.reverse()
        
        for n_rbm in copy:
            vis = gp.garray(hid)
            g_rbm = RBM(n_rbm.n_visible, n_rbm.n_hidden, n_rbm.vistype, 
                    n_rbm.hidtype, n_rbm.W, n_rbm.hbias, n_rbm.vbias)
            hid = self.get_visible(g_rbm, hid)
            gp.free_reuse_cache()
        return hid

if __name__ == "__main__":
    import GwData
    xs = GwData.GwData.as_binary()
     
    dbnetwork = DeepNet([xs.shape[1], 300, 100], ['sigmoid', 'sigmoid', 'sigmoid'], sample = False)
    dbnetwork.train(xs, [5, 50], [0.1, 0.1])
    
    out = dbnetwork.run_through_network(xs)
    echo = dbnetwork.run_down_through_network(out)
    
    mae = xs - echo
    sumSq = ((mae ** 2).sum(1) / mae.shape[1]) ** 0.5
    mae = sumSq.flatten().sum() / sumSq.shape[0]
    
    #0.0297
    pass
  