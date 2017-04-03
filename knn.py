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

    # Multiplying all distances by -1 so we can use argpartition to find the k smallest
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
    mat = np.array([[1,2,1], [0,1,0], [5,7,8], [2,4,5]])
    vec = mat [1]  # [0,1,0]

    t1 = knn(vec, mat, k=1)
    print "test 1 result:", t1
    assert set(t1) == {1}

    t2 = knn(vec, mat, k=2)
    print "test 2 result:", t2
    assert set(t2) == {0, 1}

    t3 = knn(vec, mat, k=3)
    print "test 3 result:", t3
    assert set(t3) == {0, 1, 3}


if __name__ == "__main__":
    test_knn()


