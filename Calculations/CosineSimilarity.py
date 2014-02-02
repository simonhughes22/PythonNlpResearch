import math

def cosine_similarity(a, b):
    """ Returns the cosine similarity of two vectors
        1.0 is best, -1.0 is complete inverse, 0 is orthogonal
    """

    if len(a) != len(b):
        raise Exception("Vector lengths differ")

    if len(a) == 0:
        raise Exception("One or more of the vectors has 0 length")
    
    dotProd = 0.0
    sumASq = 0.0
    sumBSq = 0.0

    for i in range(len(a)):
        dotProd += a[i] * b[i]
        sumASq  += a[i] * a[i]
        sumBSq  += b[i] * b[i]

    if sumASq == 0.0 or sumBSq == 0.0:
        return 0.0

    return dotProd / (math.sqrt( sumASq * sumBSq ))