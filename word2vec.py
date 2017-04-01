#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from gradcheck import gradcheck_naive
from sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    #Calculate the l2-norm of each row
    norms = np.sqrt(np.sum(x**2, axis=1))
    #Divide each row's elements by its norm
    x = np.divide(x, norms.reshape(-1, 1))

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    y_hat_nom = np.exp(np.dot(outputVectors, predicted))
    y_hat_denom = np.sum(y_hat_nom)
    y_hat = np.divide(y_hat_nom, y_hat_denom)
    # print("y_hat is" + str(y_hat))

    cost = -np.log(y_hat[target])

    y = np.zeros(shape=outputVectors.shape[0])
    y[target] = 1

    # print("y is" + str(y))
    # print("y_hat-y is" + str(y_hat-y))

    # gradPred = np.dot(outputVectors, y_hat-y) # todo: changed by mor
    gradPred = np.dot(y_hat - y, outputVectors)

    # Assuming predicted's shape is (d,) and y_hat and y's shape is (V,)
    # So we use the reshape function to turn them into column and row vectors so the dot product between them
    # produces a matrix, as it should.

    # grad = np.dot(predicted.reshape((-1, 1)), (y_hat-y).reshape(1, -1))
    grad = np.dot((y_hat-y).reshape(-1, 1), predicted.reshape((1, -1)))  # todo: changed by mor

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient_orig(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    indices = set(indices[1:]) #todo: added by mor
    cost = -np.log(sigmoid(np.dot(outputVectors[target], predicted)))
    cost -= np.sum([np.log(sigmoid(np.dot(-outputVectors[k], predicted))) for k in indices]) #todo: changed by mor

    # gradPred = -(1-sigmoid(np.dot(outputVectors[target], predicted))) * outputVectors[target]
    # gradPred += np.sum([(1-sigmoid(np.dot(-outputVectors[k], predicted))) * outputVectors[k] for k in indices])

    gradPred = (sigmoid(np.dot(outputVectors[target], predicted)) - 1) * outputVectors[target] #todo: changed by mor
    for k in indices:
        gradPred -= (sigmoid(np.dot(-1*outputVectors[k], predicted)) - 1) * outputVectors[k]

    grad = np.zeros(shape=outputVectors.shape)
    grad[target] = (sigmoid(np.dot(outputVectors[target], predicted)) - 1)*predicted # todo: changed by mor
    for k in indices:
        grad[k] = -1 * (sigmoid(np.dot(-1*outputVectors[k], predicted)) - 1)*predicted

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                                   K=10):
    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    sigmoidTargetPred = sigmoid(outputVectors[target, :].transpose().dot(predicted))
    cost = -np.log(sigmoidTargetPred)
    gradPred = (sigmoidTargetPred - 1.0) * outputVectors[target, :]
    grad = np.zeros(outputVectors.shape)
    grad[target, :] = predicted * (sigmoidTargetPred - 1.0)

    for s in indices:
        sigmoidSamplePredicted = sigmoid(-outputVectors[s, :].transpose().dot(predicted))
        cost -= np.log(sigmoidSamplePredicted)
        gradPred += (1.0 - sigmoidSamplePredicted) * outputVectors[s, :]


        grad[s, :] += (1.0 - sigmoidSamplePredicted) * predicted.transpose()
    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    # TODO: added by mor
    predicted = inputVectors[tokens[currentWord]]

    for i in range(len(contextWords)):  # todo: changed by mor from 2*c to len(contextWords)
        target = tokens[contextWords[i]]
        current_cost, current_gradIn, current_gradOut = \
            word2vecCostAndGradient(predicted, target, outputVectors, dataset) # todo: added by mor
        cost, gradIn[tokens[currentWord]], gradOut = cost+current_cost, gradIn[tokens[currentWord]]+current_gradIn, gradOut+current_gradOut # todo: changed by mor

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #     dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #     dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
    #     negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()