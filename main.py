# MAIN FILE, Run this to run both shallow and deep learning approaches and 
# output all performance results (plotting functions are not run but all
# plots are shown in the report)
import pandas as pd

# Code for cleaning data 
from preprocessing import cleanData
# Shallow learning code
from shallowLearning import runNaiveBayes
# Deep learning code
from deepLearning import run_lstm, run_rnn

from matplotlib import pyplot as plt

def present_results():

    # Read and clean data
    df = pd.read_csv('news_ds.csv', encoding="utf-8")
    clean_df = cleanData(df)

    print("\n----------------- Shallow Learning Results ------------------\n\n")

    print("------------ tf feature results on test set: -------------------\n")

    runNaiveBayes(clean_df, 0.02, True, True, 2)

    print("\n-------------- tf-idf feature results on test set: -------------------\n")

    runNaiveBayes(clean_df, 0.02, True, False, 2)

    print("\n--------------------------------------------------------------")


    print("----------------- Deep Learning Results ----------------------\n\n")

    print("----------------------- LSTM -----------------------------------\n\n")

    run_lstm(clean_df)

    print("\n\n------------------- RNN -----------------------------------\n\n")

    run_rnn(clean_df)

    print("\n\n------------------- END OF RESULTS -----------------------------------\n\n")
   

present_results()