import logging
import numpy as np

def __column_count__(gensim_sparse_matrix):
    return max(
                [   max([col for col,v in r]) 
                    for r in gensim_sparse_matrix
                        if len(r) > 0]
               ) + 1

def gensim_to_python_mdarray(gensim_sparse_matrix, num_columns):

    logging.log(logging.INFO, "Desparsifying distance_matrix with %i rows and %i columns", len(gensim_sparse_matrix), num_columns)

    """ 
        Takes a sparse distance_matrix (list of list of tuples)
            each tuple is an index and a value. Non-zeros are ommitted
    """

    full_matrix = []
    index = -1
    allZeros = [0 for i in range(0, num_columns)]

    for row in gensim_sparse_matrix:
        index += 1

        if len(row) == 0:
            full_matrix.append(allZeros[:])
            continue
            
        newL = []
        full_matrix.append(newL)
 
        current = 0
        for tpl in row:
            while tpl[0] > current and tpl[0] < num_columns:
                newL.append(0)
                current += 1
            newL.append(tpl[1])
            current += 1
           
        lastIndex = row[-1][0]
        while lastIndex < num_columns -1:
            newL.append(0)
            lastIndex += 1
    return full_matrix

def gensim_to_numpy_array(gensim_sparse_matrix,  num_columns = None, initial_value = None, map_fn = lambda x : x):
    
    if num_columns == None:
        num_columns = __column_count__(gensim_sparse_matrix)
    
    shape = (len(gensim_sparse_matrix), num_columns)
    arr = np.zeros(shape)
    if initial_value != None:
        arr = arr + initial_value
    
    for iRow,row in enumerate(gensim_sparse_matrix):
        for iCol,val in row:
            arr[iRow,iCol] = map_fn(val)
    
    return arr

def numpy_to_gensim_format(items):
    return [list(enumerate(i)) for i in items]

def map_matrix(mapping_fn, distance_matrix):
    r,c = distance_matrix.shape
    
    new_matrix = np.zeros(distance_matrix.shape)
    for row in range(0, r):
        for col in range(0,c):
            new_matrix[row,col] = mapping_fn(distance_matrix[row,col])
    return new_matrix

def normalize_rows(m):
    """  """
    means = np.mean(m, axis = 1, dtype=float, keepdims=True)
    sds = np.std(m, axis = 1, dtype=float, keepdims=True)
    return (m - means) / sds

def normalize_columns(m):
    """  """
    means = np.mean(m, axis = 0, dtype=float, keepdims=True)
    sds = np.std(m, axis = 0, dtype=float, keepdims=True)
    return (m - means) / sds

def make_rows_unit_vectors(m):
    """ Makes every row a unit vector """
    row_lengths = np.sqrt(np.sum(np.square(m), axis = 1, dtype=float, keepdims=True))
    return m / row_lengths

def make_cols_unit_vectors(m):
    """ Makes every row a unit vector """
    col_lengths = np.sqrt(np.sum(np.square(m), axis = 0, dtype=float, keepdims=True))
    return m / col_lengths
    
""" Vector utils """
def vector_length(vector):
    return np.sqrt(np.sum(np.square(vector)))

def unit_vector(vector):
        
    veclen = vector_length(vector)
    if veclen > 0.0:
        return vector / veclen

if __name__ == "__main__":
    
    print "Test Matrix Mapper"
    distance_matrix = np.array([[1,2,3,0,3,0], [0,0,1,2,0,10]])
    
    def as_binary(val):
        if val > 0:
            return 1
        return 0
    
    print distance_matrix
    print ""
    print map_matrix(as_binary, distance_matrix)
    