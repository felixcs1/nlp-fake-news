import numpy as np

from time import time

import pandas as pd

from preprocessing import cleanData

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, SimpleRNN, Dropout, Input, Bidirectional, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.optimizers import RMSprop

import keras.backend as K

from sklearn import metrics
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt


# The path to the glove.6B.300d.txt file containing glove vectors
# Change if appropriate
######################################################################
GLOVE_PATH = "glove.6B/glove.6B.300d.txt"
######################################################################


def plot_train_history(history):
    plt.subplot(121)
    # summarize history for accuracy
    plt.plot(history.history['acc'], 'C1')
    plt.plot(history.history['val_acc'], 'C2')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(122)
    # summarize history for loss
    plt.plot(history.history['loss'], 'C3')
    plt.plot(history.history['val_loss'], 'C4')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('history.png')
    

##############  Function for Part C i) ###########################
def get_glove_vectors(word_index):
    """

        Takes a dictionary of all unique tokens in the dataset and 
        returns a list of corresponding word2vec GloVe vectors
    """

    #load GloVe vectors
    embeddings_index = dict()
    f = open(GLOVE_PATH, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    nb_words = len(word_index)+1
    
    # create a weight matrix for words in training articles
    embedding_matrix = np.zeros((nb_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def run_lstm(df):
    """
        Run LSTM on dataset (df) and output metrics and confusion matrix
    """

    EMBEDDING_DIM = 300 # length of GloVe vectors
    MAX_SEQUENCE_LENGTH = 1000
     
    X_train, X_test, y_train, y_test = train_test_split(df.TEXT, df.LABEL, test_size=0.25, random_state=33)

    # Covert each article into a list of token indices 
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(df.TEXT)

    sequences_1 = tokenizer.texts_to_sequences(X_train)
    sequences_2 = tokenizer.texts_to_sequences(X_test)

    # Get indexed dictionary of all unique tokens in the data
    word_index =  tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    # Pad sequences with zeros so all are of a fixed length of 1000
    data_train = sequence.pad_sequences(sequences_1, maxlen=1000)
    data_test = sequence.pad_sequences(sequences_2, maxlen=1000)

    print("Sequences padded")
    # Convert data lables to numpy arrays (to input to keras)
    labels_train = np.array(y_train)
    labels_test = np.array(y_test)

    # Prepare embedding layer 
    nb_words = len(word_index)+1
    embedding_matrix = get_glove_vectors(word_index)
    
    print("Vectors obtained")
    ################ Define the layers ########################
    model = Sequential()
    model.add(Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
    )
    model.add(Dropout(0.4))
    model.add(LSTM(90, recurrent_dropout=0.4, bias_initializer='RandomNormal'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy',])
    print(model.summary())
    ############################################################

    ############## Train and evaluate the model ##########################
    start = time()
    history = model.fit(data_train, labels_train, validation_split=0.1, epochs=50, batch_size=850)
    end = time()
    print('Total time passed training: {} seconds'.format(end - start))

    predicted = model.predict_classes(data_test, verbose=1)

    print()
    print(classification_report(y_test, predicted,  labels=[0,1], target_names=["FAKE NEWS", "REAL NEWS"]))

    print("\nAccuracy: %.3f" % metrics.accuracy_score(y_test, predicted))
    print("\nF measure: %.3f" % metrics.f1_score(y_test, predicted))
    print("\nPrecision: %.3f" % metrics.precision_score(y_test, predicted))
    print("\nRecall: %.3f" % metrics.recall_score(y_test, predicted))
    print("\nConfusion Matrix: \n")
    print(metrics.confusion_matrix(y_test, predicted))

    # Plots shown in report
    # plot_train_history(history)


def run_rnn(df):
    """
        Run RNN on dataset (df) and output metrics and confusion matrix
    """

    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 1000
       
    X_train, X_test, y_train, y_test = train_test_split(df.TEXT, df.LABEL, test_size=0.25, random_state=33)

    # Covert each article into a list of token indices 
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(df.TEXT)

    sequences_1 = tokenizer.texts_to_sequences(X_train)
    sequences_2 = tokenizer.texts_to_sequences(X_test)

    # Get indexed dictionary of all unique tokens in the data
    word_index =  tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    # Pad sequences with zeros so all are of a fixed length of 1000
    data_train = sequence.pad_sequences(sequences_1, maxlen=1000)
    data_test = sequence.pad_sequences(sequences_2, maxlen=1000)

    print("Sequences padded")
    # Convert data lables to numpy arrays (to input to keras)
    labels_train = np.array(y_train)
    labels_test = np.array(y_test)

    # Prepare embedding layer 
    nb_words = len(word_index)+1
    embedding_matrix = get_glove_vectors(word_index)
    
    print("Vectors obtained")
    ################ Define the layers ########################
    model = Sequential()
    model.add(Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=False)
    )
    model.add(Dropout(0.4))
    model.add(SimpleRNN(90, recurrent_dropout=0.4, bias_initializer='RandomNormal'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy',])
    print(model.summary())
    ############################################################

    ############## Train and evaluate the model ##########################
    start = time()
    history = model.fit(data_train, labels_train, validation_split=0.1, epochs=50, batch_size=850)
    end = time()
    print('\nTotal time passed training: {} seconds'.format(end - start))

    predicted = model.predict_classes(data_test, verbose=1)
    print()
    print(classification_report(y_test, predicted,  labels=[0,1], target_names=["FAKE NEWS", "REAL NEWS"]))

    print("\nAccuracy: %.3f" % metrics.accuracy_score(y_test, predicted))
    print("\nF measure: %.3f" % metrics.f1_score(y_test, predicted))
    print("\nPrecision: %.3f" % metrics.precision_score(y_test, predicted))
    print("\nRecall: %.3f" % metrics.recall_score(y_test, predicted))
    print("\nConfusion Matrix: \n")
    print(metrics.confusion_matrix(y_test, predicted))

    #plits shown in report
    #plot_train_history(history)