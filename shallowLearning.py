import csv 
import os
# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate


from sklearn.feature_extraction import stop_words

from sklearn.metrics import classification_report
from sklearn import metrics

from langdetect import detect

import matplotlib.pyplot as plt

# Import script for cleaning data
from preprocessing import *


def runNaiveBayes(data, alpha, run_test, use_idf, ngram_num):
    """
        Runs multinomial naive bayes classifier

        data:      cleaned data frame containing all data
        alpha:     parameter to pass to MultinomialNB()
        run_test:  if true evaluate on test set, else perform 10 fold cross validation
        use_idf:   if true use tf-idf features, else just use tf
        ngram_num: the value of n for ngrams

        returns: f1-measure for to be used for the plotting functions
    """

    # Split data into training and test data (df.TEXT[:num] to take less data)
    X_train, X_test, y_train, y_test = train_test_split(data.TEXT, data.LABEL, test_size=0.25, random_state=33)

    # Extract numerical features, excluding stopword 
    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english', ngram_range=(ngram_num, ngram_num), analyzer='word', max_df=1.0, min_df=1)
    X_train_counts = count_vect.fit_transform(X_train)
    
    # Use either tf or tf-idf
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train naive bayes classifier
    clf = MultinomialNB(alpha = alpha)

    # Perform cross validation for parameter tuning
    if not run_test:
        scoring = ['precision', 'recall', 'f1']
        scores = cross_validate(clf, X_train_tfidf, y_train, scoring=scoring, cv=10, return_train_score=False)
   
        precision_val = np.mean(scores['test_precision'])
        precision_val = np.mean(scores['test_recall'])
        f1_val = np.mean(scores['test_f1'])


        return(f1_val)

    else:

        # Run classifier on test set
        clf.fit(X_train_tfidf, y_train)
        
        # ### TEST ####
        X_new_counts = count_vect.transform(X_test)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
    
        # Output results
        print(classification_report(y_test, predicted,  labels=[0,1], target_names=["FAKE NEWS", "REAL NEWS"]))
        print("\nAccuracy: %.3f" % metrics.accuracy_score(y_test, predicted))
        print("\nF measure: %.3f" % metrics.f1_score(y_test, predicted))
        print("\nPrecision: %.3f" % metrics.precision_score(y_test, predicted))
        print("\nRecall: %.3f" % metrics.recall_score(y_test, predicted))
        print("\nConfusion Matrix: \n")
        print(metrics.confusion_matrix(y_test, predicted))

    return metrics.f1_score(y_test, predicted)


def plot_tf_vs_tfidf(data_frame):

    """
        Plots the f1-measure as a fucntion of alpha for tf and tf-idf features.
        Shown in report
    """
    fs_tf = []
    fs_tfidf = []
    alphas = np.linspace(0, 0.4, 20)
    for i in alphas:
        fs_tf.append(runNaiveBayes(data_frame, i, False, False, 1))
        fs_tfidf.append(runNaiveBayes(data_frame, i, False, True, 1))

    line_tf, = plt.plot(alphas, fs_tf, label="tf")
    line_tfidf, = plt.plot(alphas, fs_tfidf, label="tf-idf")
    plt.legend(handles=[line_tf, line_tfidf])
    plt.ylabel('F-Measure')
    plt.xlabel('Alpha')
    plt.show()


def plot_ngrams(data_frame):
    """
        Plots the f1-measure for different sized ngrams,
        Shown in report
    """
    fs = []
    n = [1,2,3,4]

    for i in n:
        fs.append(runNaiveBayes(data_frame, 0.02, False, True, i))

    plt.bar(n, fs)
    plt.ylabel('F-Measure')
    plt.xticks(n, ('unigram', 'bigram', 'trigram', 'four-gram'))
    plt.show()
 