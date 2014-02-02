'''
Implementation of the Sparse Autoencoder algorithm.
    (based on Ali's matlab implementation)

@author Juan Batiz-Benet <jbenet@cs.stanford.edu>

Note:
    I found this matlab to numpy ref extremely helpful:
    http://mathesaurus.sourceforge.net/matlab-numpy.html
'''

import sys
import time
import numpy as np
from numpy import matlib
from numpy.matlib import multiply
from numpy.matlib import power

def printProgress(progress, start_time):
    elapsed = time.time() - start_time

    if progress > 0:
        total_eta_s = elapsed / progress
        total_eta_m = total_eta_s / 60
        total_eta_s = total_eta_s % 60
    else:
        total_eta_m = 0
        total_eta_s = 0

    if progress == 1:
        print
        print 'Done!'
    else:
        print ('\rProgress: %02.02f%% (Total estimated time: %dm%ds)' %
                     (float(progress) * 100.0, total_eta_m, total_eta_s))
    sys.stdout.flush()

def printErrors(errors):
    
    print "RMSE: {0}".format(errors)

class SparseAutoencoder(object):
    '''Implements the Sparse Autoencoder algorithm.'''

    # Autoencoder parameters
    LR_BT = 5

    RHO = -0.996
    ALPHA = 0.007 #Learning Rate
    LAMBDA = 0.002
    ITERATIONS = 1e3

    @staticmethod
    def from_file(fileName, trainingSet):
        sa = SparseAutoencoder(trainingSet)
        sa.__init_projector(fileName)
        return sa

    def __init__(self, trainingSet, hidden = None, num_batches = 100, outputFile=None):

        self.FEATURES = len(trainingSet[0])
        self.HIDDEN = hidden
        if self.HIDDEN == None:
            self.HIDDEN = int(2.0 / 3.0 * self.FEATURES)
        self.NUM_BATCHES = num_batches
        
        self.training_index = -1
        self.trainingSet_ = trainingSet
        if outputFile != None:
            self.__init_learner(outputFile)

    def __init_projector(self, inputFile):

        handle = open(inputFile, "r+")

        # initialize weights
        self.weights1_ = matlib.zeros((self.HIDDEN, self.FEATURES))
        self.weights1_ = self.weights1_ / matlib.sqrt(self.FEATURES)


        for r, line in enumerate(handle.readlines()):
            for c, col in enumerate(line.strip().split(" ")):
                self.weights1_[r, c] = float(col)

        handle.close()
        # initialize bias
        self.bias1_ = matlib.zeros((self.HIDDEN,))

    def __init_learner(self, outputFile):

        self.outputFile_ = outputFile

        # initialize weights
        self.weights1_ = matlib.rand(self.HIDDEN, self.FEATURES)
        self.weights1_ = self.weights1_ / matlib.sqrt(self.FEATURES)

        self.weights2_ = matlib.rand(self.FEATURES, self.HIDDEN)
        self.weights2_ = self.weights2_ / matlib.sqrt(self.HIDDEN)

        # initialize bias
        self.bias1_ = matlib.zeros((self.HIDDEN,))
        self.bias2_ = matlib.zeros((self.FEATURES,))

        # initialize rho estimate vector
        self.rho_est_ = matlib.zeros((self.HIDDEN,)).T
        self.errors = 0.0

    def trainingExample(self, index):
        '''Returns a training example.'''
        sample = self.trainingSet_[index]
        return matlib.distance_matrix(sample).T

    def run(self, epochs):
        start_time = time.time()
        printProgress(0.0, start_time)
        num_examples = len(self.trainingSet_)

        batch_size = int(round(num_examples / self.NUM_BATCHES))
        for i in xrange(0, epochs):
            self.errors = 0.0
            
            for j in range(self.NUM_BATCHES):
                batch = [self.trainingSet_[j * batch_size:j * batch_size + batch_size]][0]
                self.iteration_(matlib.distance_matrix(batch))    
            
            xs = matlib.distance_matrix(self.trainingSet_)
            output = self.feedForward_(xs)

            err = xs - output
            meanSq = np.sum(multiply(err, err),1) / err.shape[1]
            rmse = np.asarray(meanSq.T[0])[0] ** 0.5
            rmse = np.sum(rmse) / len(rmse)
            self.errors = rmse

            printProgress(float(i) / epochs, start_time)
            printErrors(self.errors)
            self.output()

        printProgress(1, start_time)

    def output(self):
        '''Outputs the current weights to the outputFile'''
        with open(self.outputFile_, 'w') as f:
            for row in xrange(0, self.weights1_.shape[0]):
                for col in xrange(0, self.weights1_.shape[1]):
                    f.write('%f ' % self.weights1_[row, col])
                f.write('\n')
    
    def __flatten_matrix__(self, distance_matrix):
        ar = np.asarray(distance_matrix, float)
        return [item[0] for item in ar]

    def iteration_(self, xs):
        '''Runs one iteration of the Sparse Autoencoder Algorithm.'''
        
        self.feedForward_(xs)
        self.backPropagate_()
        self.gradientDescent_()
        self.rhoUpdate_()

    def feedForward_(self, xs):
        '''FeedForward pass (computing activations).'''
        self.xs__ = xs

        # hidden layer
        self.z2__ = self.xs__ * self.weights1_.T + self.bias1_
        self.a2__ = matlib.tanh(self.z2__)
        
        # output layer
        self.z3__ = self.a2__ * self.weights2_.T + self.bias2_
        self.a3__ = matlib.tanh(self.z3__)
        return self.a3__;

    def backPropagate_(self):
        '''Back-Propagate errors (and node responsibilities).'''
        errors3 = multiply(-(self.xs__ - self.a3__), 1 - power(self.a3__, 2))
        d3 = multiply(errors3, self.xs__)
        
        errors2 = multiply(d3 * self.weights2_, 1 - power(self.a2__, 2))
        d2 = multiply(errors2, self.a2__)
        # average the errors
        d3 = np.sum(d3,0) / d3.shape[0]
        d2 = np.sum(d2,0) / d2.shape[0]
        
        self.d3__ = d3.T
        self.d2__ = d2.T

    def gradientDescent_(self):
        '''Gradient Descent (updating parameters).'''
        self.weights1_ = self.weights1_ - (self.ALPHA *
            (self.d2__ + self.LAMBDA * self.weights1_))
        self.bias1_ = self.bias1_ - self.ALPHA * self.d2__.T

        self.weights2_ = self.weights2_ - (self.ALPHA *
            (self.d3__ + self.LAMBDA * self.weights2_))
        self.bias2_ = self.bias2_ - self.ALPHA * self.d3__.T

    def rhoUpdate_(self):
        
        mean_a2 = np.sum(self.a2__, 0) / self.a2__.shape[0]
        
        '''Updating running rho estimate vector.'''
        self.rho_est_ = 0.999 * self.rho_est_ + 0.001 * mean_a2.T

        # updating hidden layer intercept terms based on rho estimate vector
        self.bias1_ = self.bias1_ - (self.ALPHA *
            self.LR_BT * (self.rho_est_.T - self.RHO))

    def projectTrainingSet(self):
        output = []
        for j in range(0, len(self.trainingSet_)):
            x = self.trainingExample(j)
            activated = self.projectHiddenLayer(x)
            output.append(self.__flatten_matrix__(activated))
        return output

    def projectHiddenLayer(self, x):

        # hidden layer
        z2 = self.weights1_ * x + self.bias1_.T
        a2 = matlib.tanh(z2)
        return a2

def get_binary_data():
    import GwData
    ts = GwData.GwData().as_binary()
    return ts

def train():
    iterations = 100
    outputfile = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Results\deeplearning\layer1.txt"

    ts = get_binary_data()
    ts = ts * 2
    ts = ts - 1

    sa = SparseAutoencoder(ts, 300, 100, outputfile)
    sa.run(iterations)
    sa.output()


if __name__ == "__main__":

    # train()
    # load_from_file()
    train()