### Deep Belief Network - using tensorflow and sklearn

#Import Libraries
import sys
import numpy as np
import FileProcess as fipr
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Start counting time
start_time = time.clock()

# open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file.csv", True)
X_test, y_test = fipr.load_csv("test_file.csv", True)

y_train = y_train.flatten() 
y_test = y_test.flatten()

time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

X_test_even, y_test_even = fipr.load_csv("test_file_even.csv", True)
y_test_even = y_test_even.flatten()

# Build DBN classifier
classifier = SupervisedDBNClassification(hidden_layers_structure=[20, 20],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=100,
                                         activation_function='relu',
                                         dropout_p=0.2)

# start counting time for training
time_train_start = time.clock()

# Training
classifier.fit(X_train, y_train)

# print training time
time_train_end = time.clock()
print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

'''
# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')
'''
# start counting time for testing
time_test_start = time.clock()

# Test
y_pred = classifier.predict(X_test)
print('Testing finished.\nAccuracy: %f' % accuracy_score(y_test, y_pred))

# print testing time
time_test_end = time.clock()
print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

# perform even test prediction
y_pred_even = classifier.predict(X_test_even)
print('Testing finished.\nAccuracy of even test set: %f' % accuracy_score(y_test_even, y_pred_even))

# Calculate running time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))

