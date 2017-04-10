#!/usr/bin/env python3
from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
from progress import *


def main():
    with open('scratch/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    progress = ProgressBar(1250000*2, 2000, name='tweeds')
    counter = 1
    cooc = None
    for fn in ['datasets/train_pos_full.txt', 'datasets/train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        row.append(t)
                        col.append(t2)
                progress.draw_progress_bar()

                if not counter%20000:
                    # since such a file can contain up to 1250000 lines,
                    # we create the coo_matrix already in small intermediate
                    # steps, such that the row, col and data lists don't get
                    # too large -> memory limits!
                    data = np.ones((len(row)))
                    if cooc is None:
                        cooc = coo_matrix((data, (row, col)), shape=(vocab_size,vocab_size))
                        cooc.sum_duplicates()
                    else:
                        cooc += coo_matrix((data, (row, col)), shape=(vocab_size,vocab_size))
                        cooc.sum_duplicates()
                    del data
                    row[:] = []
                    col[:] = []
                counter += 1

    if len(row) > 0:
        cooc += coo_matrix((data, (row, col)), shape=(vocab_size,vocab_size))
        cooc.sum_duplicates()
        del data
        row[:] = []
        col[:] = []
    print('Coocurence matrix writen: max = {}, nonzero elements = {}'.format(cooc.max(), cooc.count_nonzero()))
    with open('scratch/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
