import numpy as np

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    # Calculate Norm of Input Vector
    vector_norm = np.sqrt(np.sum(vector**2))
    matrix_norms = np.sqrt(np.sum(matrix**2, axis=1))

    nominators = np.dot(matrix, vector)
    denominators = vector_norm * matrix_norms

    # Multiplying all distacnes by -1 so we can use argpartition to find the k smallest
    # elements
    distances = np.divide(nominators, denominators) * (-1)

    nearest_idx = np.argpartition(distances, k)[:k]

    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


