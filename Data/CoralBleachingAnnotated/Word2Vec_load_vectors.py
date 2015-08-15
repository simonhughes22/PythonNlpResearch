import Settings
import cPickle as pickle
import numpy as np

settings = Settings.Settings()
file = settings.data_directory + "CoralBleaching/Word2Vec_Vectors.pl"

def fact_get_vector():

    with open(file, "r+") as f:
        dct = pickle.load(f)
    shape = dct["coral"].shape

    def get_vec(word):
        if word in dct:
            return dct[word]
        else:
            lc_word = word.lower()
            if lc_word in dct:
                return dct[lc_word]
            else:
                #initialize random vector
                v = np.random.random(shape)
                norm = np.linalg.norm(v)
                return v / norm
    return get_vec

if __name__ == "__main__":

    get_vector = fact_get_vector()
    from numpy.linalg import norm

    print("Norms:")
    print(("coral", norm(get_vector("coral"))))
    print(("the", norm(get_vector("the"))))
    print(("hiad8972", norm(get_vector("hiad8972"))))
    print(("1hdfj", norm(get_vector("1hdfj"))))
    pass