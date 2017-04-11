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
undeterminable = {867:1, 868:1, 869:1, 899:1, 993:1, 1881:1, 2128:1, 3219:1,
3588:1, 3765:1, 3777:1, 3822:-1, 4213:1, 4590:1, 4948:-1, 5314:1, 5327:-1,
5871:1, 5898:1, 6150:1, 7119:1, 7120:1, 7349:1, 7398:-1, 7529:-1, 7530:-1,
8093:1, 8094:1, 8113:1, 8859:1, 9290:1, 9502:1}

print('Starting to classify test set')

with open('datasets/test_data.txt') as f:
    for line in f:
        id, tweed = line.split(',', 1)
        id = int(id)
        label = -1
        tokens = [vocab.get(t, -1) for t in tweed.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        if not tokens:
            if id in undeterminable:
                label = undeterminable[id]
            print("couldn't assign tweed '{}' a feature. Manually: {}".format(tweed.strip(), label))
        else:
            vectors = np.array([xs[t] for t in tokens])
            label = clf.predict(np.average(vectors, axis=0).reshape(1, -1))
        test_tweeds.append([id, label])


# write classifications to csv file
timestamp = '{:%Y-%m-%d_%H%M%S}'.format(datetime.datetime.now())
print('Finished classication. Saving csv file to submissions/submission_{}.csv'.format(timestamp))
if not os.path.exists('submissions/'):
    os.makedirs('submissions/')
np.savetxt('submissions/submission_{}.csv'.format(timestamp), test_tweeds, delimiter=",", fmt='%d', header='Id,Prediction', comments='')
