'''
Created on Aug 18, 2013
@author: simon.hughes

Auto-encoder implementation. Can be used to implement a denoising auto-encoder, sparse or contractive auto-encoder
'''

import numpy as np
from numpy import matlib

class AutoEncoder(object):
    '''
    classdocs
    '''

    def __init__(self, num_inputs, num_hidden, learning_rate = 0.1,
                 activation_fns = ("relu", "tanh"),
                 initial_wt_max = 0.01, weight_decay = 0.0, desired_sparsity = 0.05, sparsity_wt = 0.00,
                 w1_b1 = None, w2_b2 = None):
        '''
        num_inputs = number of inputs \ outputs
        num_hidden = size of the hidden layer
        activation_fn = activation function to use ("sigmoid" | "tanh" | "linear" | "relu")
        initial_wt_max = the initial weights will be set to random weights in the range -initial_wt_max to +initial_wt_max
        weight_decay = a regularization term to stop over-fitting. Only turn on if network converges too fast or overfits the data
        
        w1_b1 are a tuple of weight matrix 1 and bias 1
        w2_b2 are a tuple of weight matrix 2 and bias 2
            This allows weight sharing between networks
        
        '''
        """ Properties """
        self.learning_rate = learning_rate
        self.activation_fns = activation_fns
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden

        """ An auto-encoder """
        num_outputs = num_inputs
        self.num_outputs = num_outputs
        self.initial_wt_max = initial_wt_max
        self.weight_decay = weight_decay
        self.desired_sparsity = desired_sparsity
        self.sparsity_wt = sparsity_wt
        """ END Properties """
        
        if w1_b1 == None:
            self.w1 = matlib.rand((num_hidden, num_inputs)) * initial_wt_max
            self.b1 = matlib.rand((1,num_hidden)) * initial_wt_max
        else:
            self.w1 = w1_b1[0]
            self.b1 = w1_b1[1]
        
        assert self.w1.shape == (num_hidden, num_inputs)
        assert self.b1.shape == (1, num_hidden)
        
        if w2_b2 == None:
            self.w2 = matlib.rand((num_outputs, num_hidden)) * initial_wt_max
            self.b2 = matlib.rand((1, num_outputs)) * initial_wt_max
        else:
            self.w2 = w2_b2[0]
            self.b2 = w2_b2[1]
        
        assert self.w2.shape == (num_outputs, num_hidden)
        assert self.b2.shape == (1, num_outputs)

        pass

    def __ensure_np__(self, a):
        if type(a) != np.array:
            a = np.array(a)
        return a
    
    def train(self, xs, epochs = 100, batch_size = 100):
        inputs  = self.__ensure_np__(xs)
        outputs = inputs.copy()

        num_rows = inputs.shape[0]
        
        """ Number of rows in inputs should match outputs """
        assert num_rows == outputs.shape[0]
        
        """ Check outputs match the range for the activation function for the layer """
        self.__validate__(inputs, 0)
        self.__validate__(outputs, 1)
        
        num_batches = num_rows / batch_size
        if num_rows % batch_size > 0:
            num_batches += 1
        
        mse = -1.0
        mae = -1.0
        for epoch in range(epochs):

            """ Note that the error may start increasing exponentially at some point
                if so, halt training
            """

            errors = None
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                mini_batch_in = inputs[start:end]
                mini_batch_out = outputs[start:end]
                if len(mini_batch_in) == 0:
                    continue

                w1ds, w2ds, b1ds, b2ds, mini_batch_errors = self.__train_mini_batch__(mini_batch_in, mini_batch_out)
                """ Apply changes """
                self.w1 -= w1ds
                self.w2 -= w2ds
                self.b1 -= b1ds
                self.b2 -= b2ds

                if errors == None:
                    errors = mini_batch_errors
                else:
                    errors = np.append(errors, mini_batch_errors, 0 )

            mse = np.mean(np.square(errors) )
            mae = np.mean(np.abs(errors))
            print "MSE for epoch {0} is {1}".format(epoch, mse),
            print "\tMAE for epoch {0} is {1}".format(epoch, mae)
        return (mse, mae)

    def get_training_errors(self, xs):

        inputs = self.__ensure_np__(xs)
        outputs = inputs.copy()
        
        num_rows = inputs.shape[0]
        """ Number of rows in inputs should match outputs """
        assert num_rows == outputs.shape[0]
        
        """ Check outputs match the range for the activation function """
        self.__validate__(outputs)
        return self.__train_mini_batch__(inputs, outputs)
     
    def __validate__(self, inputs, layer):
        activation_fn = self.activation_fns[layer]

        min_inp = np.min(inputs)
        max_inp = np.max(inputs)
        
        if activation_fn == "sigmoid":
            self.__in_range__(min_inp, max_inp,  0.0, 1.0)
        elif activation_fn == "tanh":
            self.__in_range__(min_inp, max_inp, -1.0, 1.0)
        elif activation_fn == "relu":
            self.__in_range__(min_inp, max_inp, 0.0, np.inf)
        elif activation_fn == "linear":
            pass
        else:
            raise Exception("Unknown activation function %s" % activation_fn)
   
    def __in_range__(self, actual_min, actual_max, exp_min, exp_max):
        assert actual_min >= exp_min
        assert actual_max <= exp_max

    def hidden_activations(self, inputs):
        np_inputs_T = np.array(inputs).T
        z2 = self.__compute_z__(np_inputs_T, self.w1, self.b1)
        a2 = self.__activate__(z2, 0)
        return a2.T

    def __prop_up__(self, inputs_T, wts, bias, layer):

        """ Compute activations """
        z = self.__compute_z__(inputs_T, wts, bias)
        a = self.__activate__(z, layer)
        return (z, a)

    def prop_up(self, inputs):
        inputs_T = np.array(inputs).T
        outputs_T = inputs_T.copy()
        
        """ Compute activations """
        z2, a2 = self.__prop_up__(inputs_T, self.w1, self.b1, 0)
        z3, a3 = self.__prop_up__(a2, self.w2, self.b2, 1)

        """ errors """
        errors = (outputs_T - a3)
        return (a3.T, a2.T, errors.T)

    def feed_forward(self, inputs):
        inputs_T = np.array(inputs).T

        """ Compute activations """
        z2, a2 = self.__prop_up__(inputs_T, self.w1, self.b1, 0)
        z3, a3 = self.__prop_up__(a2, self.w2, self.b2, 1)

        return a3.T


    def __train_mini_batch__(self, input_vectors, outputs):
        rows = input_vectors.shape[0]
        inputs_T = input_vectors.T
        outputs_T = outputs.T
        
        """ Compute activations """
        z2, a2 = self.__prop_up__(inputs_T, self.w1, self.b1, 0)
        z3, a3 = self.__prop_up__(a2, self.w2, self.b2, 1)
        
        """ errors = mean( 0.5 sum squared error)  """
        assert outputs_T.shape == a3.shape
        errors = (outputs_T - a3)
         
        deriv3 = self.__derivative__(a3, 1)
        deriv2 = self.__derivative__(a2, 0)
        
        """ Note: multiply does an element wise product, NOT a dot product (Hadambard product)
            inputs_T must have same shape
        """
        delta3 = np.multiply(-(errors), deriv3) # d3 is - errors multiplied by derivative of activation function
        """ THIS IS BACK PROP OF WEIGHTS TO HIDDEN LAYER"""
        
        if self.sparsity_wt > 0.0:
            """ SPARSITY PENALTY """
            if self.activation_fns[1] != "sigmoid":
                raise Exception("This is correct if activation function is not sigmoid")
            pj = np.mean(a2, axis = 1)
            p = self.desired_sparsity
            sparsity_penalty = self.sparsity_wt * ( -p/pj + (1 - p)/(1 - pj) )
            delta2 = np.multiply( np.dot(self.w2.T, delta3) + sparsity_penalty, deriv2 )
        else:
            delta2 = np.multiply( np.dot(self.w2.T, delta3), deriv2 )

        """ Delta for weights is the dot product of the delta3 (error deltas for output) and activations for that layer"""
        frows = float(rows)
        
        w1delta = np.dot(delta2, inputs_T.T) / frows
        w2delta = np.dot(delta3, a2.T) / frows
        
        """ For each weight in the weight matrix, update it using the input activation * output delta.
            Compute a mean over all examples in the batch. 
            
            The dot product is used here in a very clever  way to compute the activation * the delta 
            for each input and hidden layer node (taking the dot product of each weight over all input_vectors 
            (adding up the weight deltas) and then dividing this by num rows to get the mean
         """
        b1delta = (np.sum(delta2, 1) / frows).T
        b2delta = (np.sum(delta3, 1) / frows).T
        
        if self.weight_decay > 0.0:
            w1ds = self.learning_rate * (w1delta + self.weight_decay * self.w1 )
            w2ds = self.learning_rate * (w2delta + self.weight_decay * self.w2 )
        else:
            w1ds = self.learning_rate * (w1delta )
            w2ds = self.learning_rate * (w2delta )
        
        b1ds = self.learning_rate * b1delta
        b2ds = self.learning_rate * b2delta
        
        """ return a list of errors (one item per row in mini batch) """
        
        """ Compute Mean errors across all training examples in mini batch """

        errors = np.nan_to_num(errors)
        return (w1ds, w2ds, b1ds, b2ds, errors.T)
   
    def __compute_z__(self, inputs, weights, bias):
        #Can we speed this up by making the bias a column vector?
        return np.dot(weights, inputs) + bias.T
    
    def __activate__(self, z, layer):
        activation_fn = self.activation_fns[layer]

        if activation_fn == "sigmoid":
            return 1/ (1 + np.exp(-z))
        elif activation_fn == "tanh":
            return np.tanh(z)
        elif activation_fn == "linear":
            return z
        elif activation_fn == "relu":
            copy = z.copy()
            copy[copy < 0] = 0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")
    
    def __derivative__(self, activations, layer):
        activation_fn = self.activation_fns[layer]

        if activation_fn == "sigmoid":
            """ f(z)(1 - f(z)) """
            return np.multiply(activations, (1 - activations))
        elif activation_fn == "tanh":
            """ 1 - f(z)^2 """
            return 1 - np.square(activations)
        elif activation_fn == "linear":
            return 1.0
        elif activation_fn == "relu":
            copy = activations.copy()
            copy[copy <= 0] = 0.0
            copy[copy >  0] = 1.0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")
        
if __name__ == "__main__":

    """
    xs = [
          [1,    0,     0.5,    0.1],
          [0,    1,     1.0,    0.5],
          [1,    0.5,   1,      0  ],
          [0,    0.9,   0,      1  ],
          [0.25, 0,     0.5,    0.1],
          [0.1,  1,     1.0,    0.5],
          [1,    0.5,   0.65,   0  ],
          [0.7,  0.9,   0,      1  ]
    ]
    """
    xs = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]

    xs = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 1]
    ]
    xs = np.array(xs)

    activation_fns = ("sigmoid", "sigmoid")

    if activation_fns[0] == "tanh":
        xs = (xs - 0.5) * 2.0

    if activation_fns[1] == "tanh" and np.min(xs.flatten()) == 0.0:
        xs = (xs - 0.5) * 2.0

    num_inputs = len(xs[0])
    num_hidden = int(round(np.log2(num_inputs)))

    """ Note that the range of inputs for tanh is 2* sigmoid, and so the MAE should be 2* """
    ae = AutoEncoder(num_inputs, num_hidden, learning_rate = 0.2,
                       activation_fns= activation_fns,
                       weight_decay=0.0, desired_sparsity=0.05, sparsity_wt=0.00)

    ae.train(xs, 5000, 1)

    xs_T = np.array(xs).T
    activations = ae.hidden_activations(xs)

    """print ""
    print ae.w1
    print ae.w2
    print ""
    print ae.hidden_activations(xs)
    print np.round(ae.hidden_activations(xs))
    print ""
    print ys
    print ae.prop_up(xs, xs)[0]
    """
    print "ys"
    print np.round(xs, 1) * 1.0
    print "predictions"

    #print np.round(ae.feed_forward(xs), 1)
    print np.round(ae.feed_forward(xs), 0)
    pass

    """ TODO
    allow different activation functions per layer. Normally hidden layer uses RELU and dropout (http://fastml.com/deep-learning-these-days/)
       don't use RELU for output layer as you cannot correct for erors (i.e. gradient is 0 for negative updates!)
    implement momentum (refer to early parts of this https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf)
    implement adaptive learning rate adjustments (see link above)
    implement DROPOUT

    """
