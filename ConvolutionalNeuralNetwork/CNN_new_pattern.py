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
X_test, y_test = fipr.load_csv("test_file.csv", True)
#y_train = y_train.flatten() 
#y_test = y_test.flatten()
time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

X_test_even, y_test_even = fipr.load_csv("test_file_even.csv", True)

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


#training_labels = mnist.make_one_hot(training_labels)

print("a sample of training label:")
print(training_labels[3])

training_labels = np.array(training_labels)

print(type(training_labels))
print(type(training_labels[0]))
print(training_labels.shape)

test_data = X_test
test_labels = y_test
#test_labels = mnist.make_one_hot(test_labels)

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

test_data_even = X_test_even
test_labels_even = y_test_even
#test_labels = mnist.make_one_hot(test_labels)

# Transform even test labels to one-hot form
labels = test_labels_even
labels = np.reshape(labels,(-1)).astype(int)
one_hot_labels = [[] for i in range(len(labels))]

for i in range(len(labels)):
  if labels[i] == [0]:
      one_hot_labels[i] = [1, 0]
  else:
      one_hot_labels[i] = [0, 1]
      
test_labels_even = one_hot_labels
test_labels_even = np.array(test_labels_even)

# Definition of CNN structure
#changed to seperate conv and max_pooling
def conv_layer(input, size_in, size_out, name='conv'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([2,2,size_in,size_out], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.01, shape=[size_out]), name='B')
    conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
    
    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias',b)
    
    return tf.nn.relu(conv + b)

def pool_layer1(input, name = 'pool'):
  with tf.name_scope(name):
  
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def pool_layer2(input, name = 'pool'):
  with tf.name_scope(name):
  
    return tf.nn.max_pool(input, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')
    
def fc_layer1(input, size_in, size_out, name='fc1'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.01, shape=[size_out]), name="B")
    
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    
    return tf.nn.relu(tf.matmul(input, w) + b)
  
def fc_layer2(input, size_in, size_out, name='fc2'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.01, shape=[size_out]), name="B")
    
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    
    return tf.matmul(input, w) + b # This is where it differs from fc1. We need no activation(ReLu) for this layer.
  

# Setup placeholders and reshape data
x = tf.placeholder(tf.float32, shape=[None, 200], name='x')
y = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
x_pattern = tf.reshape(x, [-1,4,50,1])
#tf.summary.image('input', x_pattern, 3)


# Construct network
conv1 = conv_layer(x_pattern, 1, 20, 'conv1')
pool1 = pool_layer1(conv1, 'pool1')

conv2 = conv_layer(pool1, 20, 40,'conv2')
pool2 = pool_layer2(conv2, 'pool2')

flattened = tf.reshape(pool2, [-1, 1*25*40])
fc1 = fc_layer1(flattened, 1*25*40, 10, 'fc1')

# Dropout (not necessary)
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)     #Add drop out for test

# Readout layer
logits = fc_layer2(fc1_drop, 10, 2, 'fc2')


# Training and prediction
with tf.name_scope('xent'):
  xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
  tf.summary.scalar("xent", xent)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)  # Using AdamOptimizer with learning rate 1e-4

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar("accuracy", accuracy)


sess.run(tf.global_variables_initializer())


# Construct file writer
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/mnist_Imbalance/base")
writer.add_graph(sess.graph)

# start counting time for training
time_train_start = time.clock()

# Print out accuracy
for i in range(5000):
  #batch = mnist.train.next_batch(50)
  #(batch_x, batch_y) = mnist.get_next_batch(training_data,training_labels,50)
  batch = mnist.get_next_batch(training_data,training_labels,100)

  if i%5 ==0:
    s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1], keep_prob:1.0})
    writer.add_summary(s, i)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

# print training time
time_train_end = time.clock()
print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

# start counting time for testing
time_test_start = time.clock()

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_data, y: test_labels, keep_prob: 1.0}))

# print testing time
time_test_end = time.clock()
print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_data_even, y: test_labels_even, keep_prob: 1.0}))

writer.close()

# Calculate total running time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))


