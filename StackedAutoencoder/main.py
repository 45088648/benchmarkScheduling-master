"""
Construct a Denoise Stack Auto-encoder with layer of ANN
for binary classification tasks
"""
# Import libraries
import os
import sys
import urllib
import time
import FileProcess as fipr
import numpy as np
from SDA_layers import StackedDA


 
def main():
    
    
    # open and load csv files
    time_load_start = time.clock()
    X_train, y_train = fipr.load_csv("train_file.csv", True)
    X_test, y_test = fipr.load_csv("test_file.csv", True)
    #y_train = y_train.flatten() 
    #y_test = y_test.flatten()
    time_load_end = time.clock()
    print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

    X_test_even, y_test_even = fipr.load_csv("test_file_even.csv", True)

    training_data = X_train
    training_labels = y_train

    test_data = X_test
    test_labels = y_test

    test_data_even = X_test_even
    test_labels_even = y_test_even
    
    
    # building the SDA
    sDA = StackedDA([100])

    # start counting time for training
    time_train_start = time.clock()
    print('Pre-training...')
    
    # pre-trainning the SDA
    sDA.pre_train(training_data[:1000], noise_rate=0.3, epochs=100)
    print('Training Network...')
    
    # adding the final layer
    sDA.finalLayer(training_data, training_labels, epochs=500)

    # trainning the whole network
    sDA.fine_tune(training_data, training_labels, epochs=500)

    # print training time
    time_train_end = time.clock()
    print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

    # start counting time for testing
    time_test_start = time.clock()
    
    print('Testing performance...')
    # predicting using the SDA
    y_pred = sDA.predict(test_data).argmax(1)

    # print simple precision metric to the console
    print('Accuracy:  ' + str(fipr.compute_accuracy(y_test, y_pred)))
    
    # print testing time
    time_test_end = time.clock()
    print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

    # Even set test    
    y_pred_even = sDA.predict(test_data_even).argmax(1)
    
    # print simple precision metric to the console
    print('Accuracy on EVEN set:  ' + str(fipr.compute_accuracy(y_test_even, y_pred_even)))


    return sDA
    
    
if __name__ == '__main__':
    # Start counting time
    start_time = time.clock()
    
    main()

    # Calculate total running time
    print("--- Total running time: %g seconds ---" % (time.clock() - start_time))
