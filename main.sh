#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
}

start=`date +%s`
bash build_vocab.sh
bash cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
python3 glove.py
end=`date +%s`

runtime=$((end-start))
echo "Time used:" $(convertsecs $runtime)
