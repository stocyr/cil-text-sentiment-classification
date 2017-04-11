#!/usr/bin/env python3
import pickle


def main():
    stopwords = pickle.load(open('stopwords.pkl', 'rb'))
    special_tokens = ['<user>', '<url>']
    vocab = dict()
    idx = 0
    with open('scratch/vocab_cut.txt') as f:
        for line in f:
            token = line.strip()
            # filtering: stopwords
            if token in stopwords:
                continue
            # filtering: numbers
            if any(c.isdigit() for c in token):
                continue
            # filtering: punctuation
            if all(not c.isdigit() and not c.isalpha() for c in token):
                continue
            # filtering: special tokens
            if token in special_tokens:
                continue
            vocab[token] = idx
            idx += 1

    with open('scratch/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, 3)


if __name__ == '__main__':
    main()
