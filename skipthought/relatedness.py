'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1) (Sourced from skipthought)
'''
import numpy as np
import copy
import os
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation

import tensorflow as tf

tf.app.flags.DEFINE_string("relatedness_regression_factors", "relatedness_regression_factors.csv",
                           "Name of the relatedness regression factors data file.")
tf.app.flags.DEFINE_string("relatedness_regression_targets", "relatedness_regression_targets.csv",
                           "Name of the relatedness regression targets data file.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory.")

FLAGS = tf.app.flags.FLAGS


def evaluate(seed=1234, evaltest=False):
    """
    Run experiment
    """
    print 'Preparing data...'
    os.path.join(FLAGS.data_dir, FLAGS.relatedness_regression_factors)
    X = np.genfromtxt(os.path.join(FLAGS.data_dir, FLAGS.relatedness_regression_factors))
    print(X.size)

    y = np.genfromtxt(os.path.join(FLAGS.data_dir, FLAGS.relatedness_regression_targets))
    print(y.size)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print 'Encoding labels...'
    trainY = encode_labels(y_train)
    devY = encode_labels(y_test)

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=len(X[0]))

    print 'Training...'
    bestlrmodel = train_model(lrmodel, X_train, trainY, X_test, devY, y_test)

    if evaltest:

        print 'Evaluating...'
        r = np.arange(1, 6)
        yhat = np.dot(bestlrmodel.predict_proba(X_test, verbose=2), r)
        pr = pearsonr(yhat, y_test)[0]
        sr = spearmanr(yhat, y_test)[0]
        se = mse(yhat, y_test)
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_dim=ninputs))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1, 6)

    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=2, shuffle=False,
                    validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            print score
            best = score
            bestlrmodel = copy.deepcopy(lrmodel)
        else:
            done = True

    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    print 'Dev Pearson: ' + str(score)
    return bestlrmodel


def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i + 1 == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i + 1 == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    return Y

if __name__ == '__main__':
    evaluate()
