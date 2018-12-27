import csv 
import os
import re
import string
# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pandas as pd
import numpy as np

from langdetect import detect
from langdetect import lang_detect_exception

import matplotlib.pyplot as plt

### Data cleaning functions ########################

def removeWhiteSpace(text):
    clean_text = re.sub('\s+',' ', text)
    return clean_text

def removeEmails(text):
    clean_text = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', text)
    return clean_text

def removeURLS(text):
    clean_text = re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', ' ', text)
    return clean_text

def removeDates(text):
    clean_text = re.sub(r'[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}', ' ', text)
    return clean_text

def removeHexChars(text):
    clean_text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    return clean_text

def cleanData(data_frame):

    print("\nCleaning data......\n")
    for index, row in data_frame.iterrows():

        # Clean each article
        clean_text = row.TEXT.lower()
        clean_text = removeHexChars(clean_text)
        clean_text = removeEmails(clean_text)
        clean_text = removeURLS(clean_text)
        clean_text = removeDates(clean_text)
        clean_text = removeWhiteSpace(clean_text)

        # Remove blank entries
        if (len(clean_text) == 1 or len(clean_text) == 0):
            data_frame.drop(index, inplace=True)
        else:
            data_frame.at[index, 'TEXT'] = clean_text

    print("Done \n")
    print("Shape of data frame: ", data_frame.shape)
    return data_frame