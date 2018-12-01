### Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import sys
import numpy as np
import FileProcess as fipr
import time


# Start counting time
start_time = time.clock()


# open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file_400atb.csv", True)
X_test, y_test = fipr.load_csv("test_file_400atb.csv", True)
y_train = y_train.flatten() 
y_test = y_test.flatten()
time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

X_test_even, y_test_even = fipr.load_csv("test_file_400atb_even.csv", True)
y_test = y_test.flatten()

# Create a Gaussian Classifier
model = GaussianNB()

# start counting time for training
time_train_start = time.clock()

# Train the model using the training sets 
model.fit(X_train, y_train)

# print training time
time_train_end = time.clock()
print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

# start counting time for testing
time_test_start = time.clock()

#Predict Output 
y_pred = model.predict(X_test)

# print testing time
time_test_end = time.clock()
print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

# print simple precision metric to the console
print('Accuracy:  ' + str(fipr.compute_accuracy(y_test, y_pred)))

#Predict Output 
y_pred_even = model.predict(X_test_even)

# print simple precision metric to the console
print('Accuracy on EVEN test set:  ' + str(fipr.compute_accuracy(y_test_even, y_pred_even)))

# Calculate running time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))
