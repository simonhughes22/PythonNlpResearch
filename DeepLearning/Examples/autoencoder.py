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
    rows = len(errors)
    cols = len(errors[0])

    totalColErrors = 0.0
    for c in range(cols):
        totalSquaredErrors = 0.0
        for r in range(rows):
            # Total Squared Errors
            totalSquaredErrors += errors[r][c] ** 2
        # RMSE
        totalColErrors += (totalSquaredErrors / rows) ** 0.5
    mean_col_errors = totalColErrors / cols
    print "RMSE: {0}".format(mean_col_errors)

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

    def __init__(self, trainingSet, hidden = None, outputFile=None):

        self.FEATURES = len(trainingSet[0])
        self.HIDDEN = hidden
        if self.HIDDEN == None:
            self.HIDDEN = int(2.0 / 3.0 * self.FEATURES)
        
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
        self.errors = []

    def trainingExample(self, index):
        '''Returns a training example.'''
        sample = self.trainingSet_[index]
        return matlib.distance_matrix(sample).T

    def run(self, epochs):
        start_time = time.time()
        printProgress(0.0, start_time)
        num_examples = len(self.trainingSet_)

        for i in xrange(0, epochs):
            self.errors = []
            for j in range(0, num_examples):
                x = self.trainingExample(j)
                self.iteration_(x)

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
        output = self.feedForward_(xs)

        errors = xs - output
        self.errors.append(self.__flatten_matrix__(errors))

        self.backPropagate_()
        self.gradientDescent_()
        self.rhoUpdate_()

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

    def feedForward_(self, xs):
        '''FeedForward pass (computing activations).'''
        self.xs__ = xs

        # hidden layer
        self.z2__ = self.weights1_ * self.xs__ + self.bias1_.T
        self.a2__ = matlib.tanh(self.z2__)

        # output layer
        self.z3__ = self.weights2_ * self.a2__ + self.bias2_.T
        self.a3__ = matlib.tanh(self.z3__)
        return self.a3__;

    def backPropagate_(self):
        '''Back-Propagate errors (and node responsibilities).'''
        self.d3__ = multiply(-(self.xs__ - self.a3__), 1 - power(self.a3__, 2))
        self.d2__ = multiply(self.weights2_.T * self.d3__, 1 - power(self.a2__, 2))

    def gradientDescent_(self):
        '''Gradient Descent (updating parameters).'''
        self.weights1_ = self.weights1_ - (self.ALPHA *
            (self.d2__ * self.xs__.T + self.LAMBDA * self.weights1_))
        self.bias1_ = self.bias1_ - self.ALPHA * self.d2__.T

        self.weights2_ = self.weights2_ - (self.ALPHA *
            (self.d3__ * self.a2__.T + self.LAMBDA * self.weights2_))
        self.bias2_ = self.bias2_ - self.ALPHA * self.d3__.T

    def rhoUpdate_(self):
        '''Updating running rho estimate vector.'''
        self.rho_est_ = 0.999 * self.rho_est_ + 0.001 * self.a2__

        # updating hidden layer intercept terms based on rho estimate vector
        self.bias1_ = self.bias1_ - (self.ALPHA *
            self.LR_BT * (self.rho_est_.T - self.RHO))

def get_data():
    import GwData
    import WordTokenizer
    import TermFrequency
    import MatrixHelper
    import Converter

    data = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count=5)
    tf = TermFrequency.TermFrequency(tokenized_docs)

    ts = MatrixHelper.gensim_to_numpy_array(tf.distance_matrix, None, 0, Converter.to_binary)
    return ts

def train():
    iterations = 100
    outputfile = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Results\deeplearning\layer1.txt"

    ts = get_data()
    ts = ts * 2 -1

    sa = SparseAutoencoder(ts, 600, outputfile)
    sa.run(iterations)
    sa.output()

def train_level2():
    #DOES NOT LOAD WEIGHTS!
    
    iterations = 100
    outputfile = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Results\deeplearning\layer2.txt"

    ts = load_from_file()

    sa = SparseAutoencoder(ts, 300, outputfile)
    sa.run(iterations)
    sa.output()

def load_from_file():
    inputfile = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Results\deeplearning\weights.txt"

    ts = get_data()
    sa = SparseAutoencoder.from_file(inputfile, ts)
    projected = sa.projectTrainingSet()
    return projected

def load_from_file2():
    inputfile = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Results\deeplearning\layer2.txt"

    ts = load_from_file()
    sa = SparseAutoencoder.from_file(inputfile, ts)
    projected = sa.projectTrainingSet()
    return projected

if __name__ == "__main__":

    # train()
    # load_from_file()
    train()