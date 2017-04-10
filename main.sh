#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

# helper function for timing purposes
convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
}

echo "--------------------------------------------------------------------------------"
echo "Starting text sentiment classification bash script"
echo "--------------------------------------------------------------------------------"

# if running on Euler cluster, first load python3 module
NODENAME="$(uname -n)"
if [[ ${NODENAME:0:5} = "euler" ]]; then
	echo "Running on Euler cluster! --> loading python3 module..."
	module load python/3.3.3
else
	echo "Not running on Euler cluster."
fi

# Start measuring time
start=`date +%s`

echo ""
echo "Build unique vocabulary set and count occurences  ->  store in scratch/vocab.txt"
echo "--------------------------------------------------------------------------------"
bash build_vocab.sh

echo ""
echo "Filter words which occure les than 5 times  ->  store in scratch/vocab_cut.txt"
echo "--------------------------------------------------------------------------------"
bash cut_vocab.sh

echo ""
echo "Convert text-based file to binary file  ->  store in scratch/vocab.pkl"
echo "--------------------------------------------------------------------------------"
python3 pickle_vocab.py

echo ""
echo "Compute coocurence matrix from tweeds  ->  store in cooc.pkl"
echo "--------------------------------------------------------------------------------"
python3 cooc.py

echo ""
echo "Learn word embedding with GloVe  ->  store in scratch/embeddings.pkl"
echo "--------------------------------------------------------------------------------"
python3 glove.py

echo ""
echo "Train linear classifier  ->  store in scratch/classifier.pkl"
echo "--------------------------------------------------------------------------------"
python3 train.py

echo ""
echo "Classify test data for submission  ->  save classification in folder submission/"
echo "--------------------------------------------------------------------------------"
python3 classify.py

# stop measuring time
end=`date +%s`

runtime=$((end-start))
echo "--------------------------------------------------------------------------------"
echo "Time used:" $(convertsecs $runtime)
