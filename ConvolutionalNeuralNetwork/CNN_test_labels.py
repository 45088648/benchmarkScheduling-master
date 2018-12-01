### Build CNN for Data with New Patterns ###

# Import libraries
import numpy as np
import os
import sys
import urllib
import time
import tensorflow as tf
import FileProcess as fipr

from Mnist import Mnist
mnist = Mnist()

sess = tf.InteractiveSession()


# Start counting time
start_time = time.clock()

# open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file.csv", True)
#X_test, y_test = fipr.load_csv("test_file.csv", True)
#y_train = y_train.flatten() 
#y_test = y_test.flatten()
time_load_end = time.clock()
#print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

training_data = X_train
training_labels = y_train

print(type(training_labels))
print(type(training_labels[0,0]))
print(training_labels.shape)

print('original labels:')
print(training_labels[3])

#training_labels = mnist.change_one_hot(training_labels)
labels = training_labels

labels = np.reshape(labels,(-1)).astype(int)

one_hot_labels = [[] for i in range(len(labels))]

for i in range(len(labels)):
  if labels[i] == [0]:
      one_hot_labels[i] = [1, 0]
  else:
      one_hot_labels[i] = [0, 1]

training_labels = one_hot_labels


print('one-hot labels:')
print(training_labels[3])

print(type(training_labels))
print(type(training_labels[0]))
#print(training_labels.shape)

training_labels = np.array(training_labels)
print(type(training_labels))
print(type(training_labels[0]))
print(training_labels.shape)
print('one-hot labels:')
print(training_labels[3])
      

