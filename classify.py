#!/usr/bin/env python3

import csv
import datetime
import pickle
import numpy as np
import os
from sklearn.externals import joblib
from scipy.sparse import *

# classify the vectors in the test data

xs = pickle.load(open('scratch/embeddings.pkl', 'rb'))
vocab = pickle.load(open('scratch/vocab.pkl', 'rb'))

# load classifier state from file
clf = joblib.load('scratch/classifier.joblib.pkl')

test_tweeds = []
undeterminable = {4590:1}

with open('datasets/test_data.txt') as f:
    for line in f:
        id, tweed = line.split(',', 1)
        id = int(id)
        label = -1
        tokens = [vocab.get(t, -1) for t in tweed.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        if not tokens:
            print(id)
            label = undeterminable[id]
            print("couldn't assign tweed '{}' a feature. Manually: {}".format(tweed.strip(), label))
        else:
            vectors = np.array([xs[t] for t in tokens])
            label = clf.predict(np.average(vectors, axis=0).reshape(1, -1))
        test_tweeds.append([id, label])

# write classifications to csv file
timestamp = datetime.date.today()
if not os.path.exists('submissions/'):
    os.makedirs('submissions/')
np.savetxt('submissions/submission_{}.csv'.format(timestamp), test_tweeds, delimiter=",", fmt='%d', header='Id,Prediction', comments='')
