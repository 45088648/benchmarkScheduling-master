### Use SVM model from Scikit_learn on Shop-floor spam data

# Import Library of Simple SVM model from Scikit-learn

from sklearn import datasets
from sklearn import svm
import sys
import numpy as np
import FileProcess as fipr
import time

# For test only
#digits = datasets.load_digits()
#print(digits.data)
#print(digits.target)

# Start counting time
start_time = time.clock()

# Open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file.csv", True)
X_test, y_test = fipr.load_csv("test_file.csv", True)
print(y_test)
y_train = y_train.flatten() 
y_test = y_test.flatten()
print(y_test)
time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))


# Change labels from {0, 1} to {-1, 1} for better training
for i in range(0, len(y_train)-1):
    if y_train[i] == 0:
        y_train[i] = -1

for j in range(0, len(y_test)-1):
    if y_test[j] == 0:
        y_test[j] = -1

# Form the SVM model 
model = svm.SVC(kernel='linear', C = 1.0)

# Start counting time for training
time_train_start = time.clock()

# Train the model
model.fit(X_train, y_train)

# print training time
time_train_end = time.clock()
print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

# start counting time for testing
time_test_start = time.clock()

# Predict Output 
y_pred = model.predict(X_test)

# print testing time
time_test_end = time.clock()
print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))


# print simple precision metric to the console
print('Accuracy:  ' + str(fipr.compute_accuracy(y_test, y_pred)))

# Calculate train and eval time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))
