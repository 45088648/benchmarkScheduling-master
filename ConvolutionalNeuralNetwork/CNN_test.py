### Build CNN for Experiments ###

# Import libraries
import numpy as np
import os
import sys
import urllib
import time
import tensorflow as tf
sess = tf.InteractiveSession()



from Mnist import Mnist
mnist = Mnist()
training_data,training_labels = mnist.load_training_batch()




# Uniformed Distribution of Data
training_data,training_labels = mnist.balance_data(training_data,training_labels);
print(type(training_labels))
print(type(training_labels[0,0]))
print(training_labels.shape)
mnist.print_sample_distribution(training_labels)



training_labels = mnist.make_one_hot(training_labels)

#print("a sample of training label:" % training_labels)

test_data,test_labels = mnist.load_test_batch()
test_data = np.reshape(test_data,(-1,784))
test_labels = mnist.make_one_hot(test_labels)

#changed to seperate conv and max_pooling!!!
def conv_layer(input, size_in, size_out, name='conv'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5,5,size_in,size_out], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='B')
    conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
    
    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias',b)
    
    return tf.nn.relu(conv + b)

def pool_layer(input, name = 'pool'):
  with tf.name_scope(name):
  
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
def fc_layer1(input, size_in, size_out, name='fc1'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    
    return tf.nn.relu(tf.matmul(input, w) + b)
  
def fc_layer2(input, size_in, size_out, name='fc2'):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    
    return tf.matmul(input, w) + b # This is where it differs from fc1. We need no activation(ReLu) for this layer.
  

# Setup placeholders and reshape data
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input', x_image, 3)

# Construct network
conv1 = conv_layer(x_image, 1, 32, 'conv1')
pool1 = pool_layer(conv1, 'pool1')

conv2 = conv_layer(pool1, 32, 64,'conv2')
pool2 = pool_layer(conv2, 'pool2')

flattened = tf.reshape(pool2, [-1, 7*7*64])
fc1 = fc_layer1(flattened, 7*7*64, 1024, 'fc1')

# Dropout (not necessary)
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)     #Add drop out for test

# Readout layer
logits = fc_layer2(fc1_drop, 1024, 10, 'fc2')


# Start counting time
start_time = time.clock()

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


# Print out accuracy
for i in range(10000):
  #batch = mnist.train.next_batch(50)
  batch = mnist.get_next_batch(training_data,training_labels,50)

  if i%5 ==0:
    s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1], keep_prob:1.0})
    writer.add_summary(s, i)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %.5g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

'''
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
'''
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_data, y: test_labels, keep_prob: 1.0}))

writer.close()

# Calculate running time
print("--- %s seconds ---" % (time.clock() - start_time))
