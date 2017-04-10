#!/usr/bin/env python3

from scipy.sparse import *
import numpy as np
import pickle
from progress import *

################################################
# Vocabulary Statistics (before word embedding)
################################################

# load vocabulary file as list of tuples: (word, # of occurences)
raw_vocab = []
with open('scratch/vocab.txt') as f:
    for line in f:
        raw_vocab.append((line[8:], int(line[:7])))

# filter vocabulary by number of occurence
raw_vocab_cut = [tup for tup in raw_vocab if tup[1] >= 5]

n_words_total = 0
for tup in raw_vocab:
    n_words_total += tup[1]
n_words_cut = 0
for tup in raw_vocab_cut:
    n_words_cut += tup[1]

print('number of unique words:', len(raw_vocab))
print('number of unique words with occurence >= 5:', len(raw_vocab_cut))
print('remaining fraction of tokens in dictionary:', '{:.2f}%'.format(len(raw_vocab_cut)/len(raw_vocab)*100))
print('remaining fraction of words in filtered corpus:', '{:.2f}%'.format(n_words_cut/n_words_total*100))

# filter remaining vocabulary for words with numbers in them
raw_vocab_digit = [tup for tup in raw_vocab_cut if any(c.isdigit() for c in tup[0])]

print('fraction of words containing numbers in filtered corpus:', '{:.2f}%'.format(len(raw_vocab_digit)/len(raw_vocab_cut)*100))



################################################
# Tweed Statistics (after word embedding)
################################################

xs = pickle.load(open('scratch/embeddings_new.pkl', 'rb'))
vocab = pickle.load(open('scratch/vocab.pkl', 'rb'))
vocab_inv = {vocab[w]:w for w in vocab}

# build matrix of all tweeds with two columns:
# number of tokens in the tweed and number of tokens that we could match with our dictionary

match = []

print('\nTrying to assign feature vectors to tweeds')
progress = ProgressBar(2500000, name='tweeds')
counter = 0
for fn in ['datasets/train_pos_full.txt', 'datasets/train_neg_full.txt']:
    with open(fn) as f:
        for line in f:
            total_tokens = [vocab.get(t, -1) for t in line.strip().split()]
            assignable_tokens = [t for t in total_tokens if t >= 0]
            match.append((len(total_tokens), len(assignable_tokens)))
            if counter % 10000 == 0:
                progress.draw_progress_bar(counter)
            counter += 1

npmatch = np.array(match)

print('Number of Tweeds with no tokens assigned: {:} of {}'.format(len([tup for tup in match if tup[1] == 0]), len(match)))
print('Number of Tweeds with only one token assigned: {:} of {}'.format(len([tup for tup in match if tup[1] == 1]), len(match)))
print('Number of Tweeds with all tokens assigned: {:} of {}'.format(len([tup for tup in match if tup[1] == tup[0]]), len(match)))

print('Average coverage with tokens: {:.2f}%'.format(np.average(npmatch[:,1] / npmatch[:,0])*100))
print('Median coverage with tokens: {:.2f}%'.format(np.median(npmatch[:,1] / npmatch[:,0])*100))

