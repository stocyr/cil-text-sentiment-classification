#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
if command -v pv >/dev/null 2>&1 ;then
    # pv exists
    pv -l datasets/train_pos_full.txt datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > scratch/vocab.txt
else
    # work without pv
    cat datasets/train_pos_full.txt datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > scratch/vocab.txt
fi

