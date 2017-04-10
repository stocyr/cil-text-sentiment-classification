#!/usr/bin/env python3

# import computationally expensive model from file
from scipy.sparse import *
import numpy as np
import pickle

xs = pickle.load(open('scratch/embeddings.pkl', 'rb'))
vocab = pickle.load(open('scratch/vocab.pkl', 'rb'))
vocab_inv = {vocab[w]:w for w in vocab}


# build feature vectors for every tweed in the full corpus set
training_tweeds = []

print('Assigning feature vector to tweeds for training...')
counter = 1
for label, fn in zip([1, -1], ['datasets/train_pos_full.txt', 'datasets/train_neg_full.txt']):
    with open(fn) as f:
        for line in f:
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            if not tokens:
                print("  couldn't assign tweed '{}' a feature.".format(line[:-2]))
                continue
            vectors = np.array([xs[t] for t in tokens])
            avg_vector = np.average(vectors, axis=0)
            training_tweeds.append((label,avg_vector))
            
            if counter % 50000 == 0:
                print("processed {} tweeds (working on set '{}')".format(counter, fn))
            counter += 1
print("Finished processing {} tweeds.".format(counter, fn))


# train a linear classifier with support vector machine
from sklearn import svm
from sklearn.externals import joblib
# to separate labels and feature vectors of the training data, just unzip the tweed list
Y, X = zip(*training_tweeds)
clf = svm.LinearSVC(verbose=10)

print("Start fitting the linear classifier...".format(counter, fn))
clf.fit(X,Y)

# save classifier state to file
joblib.dump(clf, 'scratch/classifier.joblib.pkl', compress=9)
