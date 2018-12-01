### Ensemlbe Classifier with new API in order to fit new pattern ###

# Import Libraries
import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import os
import sys
import urllib
import time
import FileProcess as fipr

from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Start counting time
start_time = time.clock()


# Load Data (Test)
seed = 2017
np.random.seed(seed)

'''
data = load_iris()
idx = np.random.permutation(150)
X = data.data[idx]
y = data.target[idx]
print("Iris data shape and format:")
print(type(X))
print(type(y))
print(y.shape)
'''
# open and load csv files
time_load_start = time.clock()

X_train, y_train = fipr.load_csv("train_file.csv", True)
X_test, y_test = fipr.load_csv("test_file.csv", True)

time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

X_test_even, y_test_even = fipr.load_csv("test_file_even.csv", True)

training_data = X_train
training_labels = y_train.flatten()
test_data = X_test
test_labels = y_test.flatten()

test_data_even = X_test_even
test_labels_even = y_test_even.flatten()
'''
print("Shop floor data shape and format:")
print(type(training_labels))
print(type(training_labels[0,0]))
print(training_labels.shape)
'''
## Build Ensemble CLF with Random Forest, SVC and Logistic Regression
# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)

# Build the first layer
ensemble.add([RandomForestClassifier(random_state=seed), SVC()])

# Attach the final meta estimator
ensemble.add_meta(LogisticRegression())

## Use the model for training and testing
# start counting time for training
time_train_start = time.clock()

# Fit ensemble
ensemble.fit(training_data, training_labels)

# print training time
time_train_end = time.clock()
print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

# start counting time for testing
time_test_start = time.clock()

# Predict
preds = ensemble.predict(test_data)

# print testing time
time_test_end = time.clock()
print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

print("Fit data:\n%r" % ensemble.data)

print("Prediction score: %.6f" % accuracy_score(preds, test_labels))

# Predict
preds_even = ensemble.predict(test_data_even)

print("Fit data:\n%r" % ensemble.data)

print("Prediction score: %.6f" % accuracy_score(preds_even, test_labels_even))

