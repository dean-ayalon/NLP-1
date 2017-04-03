from utils.glove import loadWordVectors
from utils.treebank import StanfordSentiment
from knn import *

# Loading original dataset and tokens
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
wordVectors = loadWordVectors(tokens)

# Calculating nearest-neighbours word lists
key_words = ["the", "unique", "superb", "comedy", "surprisingly"]
inputVectors = wordVectors[:nWords]
inv_tokens = {v: k for k, v in tokens.iteritems()}
for key_word in key_words:
    wordVector = inputVectors[tokens[key_word]]
    idx = knn(wordVector, inputVectors, 11)
    print "Words related to \"" + key_word + "\": ",  [inv_tokens[i] for i in idx]

