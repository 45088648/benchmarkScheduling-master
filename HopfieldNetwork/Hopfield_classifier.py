"""
Train a Hopfield network to classify MNIST digits.

Since Hopfield networks are not supervised models, we must
turn classification into a memory recall task. To do this,
we feed the network augmented vectors containing both the
image and a one-hot vector representing the class.
"""
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

from update import hebbian_update
from update import extended_storkey_update
from network import Network
# from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

# Start counting time
start_time = time.clock()

# open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file.csv", True)
X_test, y_test = fipr.load_csv("test_file.csv", True)

time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

training_data = X_train
training_labels = y_train

# Transform training labels to one-hot form
labels = training_labels
labels = np.reshape(labels,(-1)).astype(int)
one_hot_labels = [[] for i in range(len(labels))]

for i in range(len(labels)):
  if labels[i] == [0]:
      one_hot_labels[i] = [1, 0]
  else:
      one_hot_labels[i] = [0, 1]
      
training_labels = one_hot_labels
training_labels = np.array(training_labels)

test_data = X_test
test_labels = y_test


# Transform testing labels to one-hot form
labels = test_labels
labels = np.reshape(labels,(-1)).astype(int)
one_hot_labels = [[] for i in range(len(labels))]

for i in range(len(labels)):
  if labels[i] == [0]:
      one_hot_labels[i] = [1, 0]
  else:
      one_hot_labels[i] = [0, 1]
      
test_labels = one_hot_labels
test_labels = np.array(test_labels)




def main():
    """
    Train the model and measure the results.
    """
    network = Network(200 + 2)
    with tf.Session() as sess:
        print('Training...')
        # start counting time for training
        time_train_start = time.clock()
        train(sess, Network, training_data, training_labels)

        # print training time
        time_train_end = time.clock()
        print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))
        print('Evaluating...')
        # print('Validation accuracy: ' + str(accuracy(sess, network, MNIST.validation)))
        
        #print('Training accuracy: ' + str(accuracy(sess, network, training_data, training_labels)))
        # start counting time for testing
        time_test_start = time.clock()

        # print testing time
        time_test_end = time.clock()
        print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

        print('Testing accuracy: ' + str(accuracy(sess, network, test_data, test_labels)))

def train(sess, network, data, label):
    """
    Train the Hopfield network.
    """
    data_ph = tf.placeholder(tf.float32, shape=(200,))
    label_ph = tf.placeholder(tf.float32, shape=(2,))
    #joined = tf.concat((tf.greater_equal(data_ph, 0.5), label_ph), axis=-1)
    joined = tf.concat((data_ph, label_ph), axis=-1)
    #update = hebbian_update(joined, network.weights)
    update = extended_storkey_update(joined, network.weights)
    sess.run(tf.global_variables_initializer())
    i = 0
    
    for label, data in zip(label, data):
        sess.run(update, feed_dict={data_ph: data, label_ph: label})
        i += 1
        '''if i % 1000 == 0:
            elapsed = time.time() - start_time
            frac_done = i / len(dataset.images)
            remaining = elapsed * (1-frac_done)/frac_done
            print('Done %.1f%% (eta %.1f minutes)' % (100 * frac_done, remaining/60))'''

def accuracy(sess, network, data, label):
    """
    Compute the test-set accuracy of the Hopfield network.
    """
    data_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 200))
    preds = classify(network, data_ph)
    num_right = 0
    for i in range(0, len(label), BATCH_SIZE):
        data_batch = data[i : i+BATCH_SIZE]
        labels_batch = label[i : i+BATCH_SIZE]
        preds_out = sess.run(preds, feed_dict={data_ph: data_batch})
        num_right += np.dot(preds_out.flatten(), labels_batch.flatten())
    return num_right / len(label)

def classify(network, data):
    """
    Classify the images using the Hopfield network.

    Returns:
      A batch of one-hot vectors.
    """
    #numeric_vec = tf.cast(tf.greater_equal(data, 0.5), tf.float32)*2 - 1
    numeric_vec = data
    thresholds = network.thresholds[-2:]
    logits = tf.matmul(numeric_vec, network.weights[:200, -2:]) - thresholds
    return tf.one_hot(tf.argmax(logits, axis=-1), 2)

if __name__ == '__main__':
    main()

# Calculate total running time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))
