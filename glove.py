#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
from progress import *
import sys


def glove():
    print("loading cooccurrence matrix")
    with open('scratch/cooc.pkl', 'rb') as f:
        csr = pickle.load(f)
    print("nonzero entries: {}".format(csr.nnz))
    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", csr.max())

    #prg = ProgressBar(len(csr.data), 100000, 'entries')
    #for i,d in enumerate(csr.data):
    #    csr.data[i] = min(d, nmax)
    #    prg.draw_progress_bar()
    #cooc = coo_matrix(csr, dtype=np.uint8)
    cooc = coo_matrix(csr)

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10
    
    f = lambda n : min(1, (n/nmax)**alpha)

    print("starting SGD")
    progress = ProgressBar(epochs*len(cooc.data), name='iters')
    counter = 1
    for epoch in range(epochs):
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            if not counter % 5000:
                progress.draw_progress_bar(counter)
            counter += 1
            #sys.stdout.write(" (epoch {})".format(epoch))
            xs[ix] = xs[ix] + ( 2*eta*f(n)*(np.log(n) - xs[ix].dot(ys[jy])) ) * ys[jy]
            ys[jy] = ys[jy] + ( 2*eta*f(n)*(np.log(n) - xs[ix].dot(ys[jy])) ) * xs[ix]    
    with open('scratch/embeddings.pkl', 'wb') as f:
        pickle.dump(xs, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    glove()
