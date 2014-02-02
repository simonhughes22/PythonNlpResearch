from sklearn.cluster import KMeans
import logging

class Clusterer(object):
    """ 
        Run the clustering algorithm
    """
    def __init__(self,k):
        self.algorithm = KMeans(k = k)
        self.k = k

    def Run(self, distance_matrix):
        logging.log(logging.INFO, "Running clustering algorithm with k of {0}".format(self.k))
        
        self.algorithm.fit(distance_matrix)
        return self.algorithm.labels_[:]