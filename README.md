text sentiment classification
=============================

The use of microblogging and text messaging as a media of communication has greatly increased over the past 10 years. Such large volumes of data amplifies the need for automatic methods to understand the opinion conveyed in a text.

Resources
---------
All the necessary resources (including training data) are available at https://inclass.kaggle.com/c/cil-text-classification

Training Data
-------------
For this problem, we have acquired 2.5M tweets classified as either positive or negative.

Evaluation Metrics
------------------
Your approach is evaluated according to the following criteria:

- Classification Accuracy


processing chain
=================

1. build_vocab.sh
   build unique vocabulary set and count occurences
   -> here we select whether the small or the full corpus are used
   Time usage: tbd
   File size: ~ 20 MB

2. cut_vocab.sh
   filter out words with occurences < 5
   Time usage: tbd
   File size: ~ 1.5 MB

3. python3 pickle_vocab.py
   store filtered vocabulary set in binary file vocab.pkl
   Time usage: tbd
   File size: ~ 2.5 MB

4. python3 cooc.py
   -> here we also have to select the small or full corpus
   compute coocurence matrix and store in binary file cooc.pkl
   Time usage: 7min
   File size: ~ 800 MB

5. Word embedding using GloVe
   store word embedding in binary file embeddings.pkl
   Time usage: 1.5h
   File size: ~ 35 MB

6. python3 train.py
   Building feature vectors from training dataset and Train linear classifier (logistic regression or SVM?)
   store classifier state in binary file classifier.joblib.pkl
   Time usage: ~0.5h

7. python3 classify.py
   Classify test data
   store classified submission file to submission_<timestamp>.csv

8. [optionally] classify small validation set (EXCLUDED FROM TRAINING SET!) for faster accuracy report

9. [optionally] automatcally build submission comment with current parameter setup for csv file submission

__(file sizes and computation times are listed using the full corpus of 2*1.25M tweeds)__