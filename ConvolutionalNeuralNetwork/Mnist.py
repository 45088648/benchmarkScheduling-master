### MNIST Implementation for Hypothetical Data of Shop-floor###
# MNist Notes: 
# Labels go from 0-9. 
# Total number of samples: training: 60000, test: 10000.
# samples per class are imbalanced
'''
Samples per class(0-9):
5923
6742
5958
6131
5842
5421
5918
6265
5851
5949
'''
import numpy as np
import os
import sys
python_version = int(sys.version[0]);
if(python_version == 3):
    # For Python 3.0 and later
    from urllib.request import urlretrieve
else:
    # Python 2's urllib
    import urllibimport 
import gzip
import struct
from struct import unpack

import random
# For visualization of data
from PIL import Image
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from skimage import transform as tf
from skimage.transform import rotate
import sklearn.preprocessing


class Mnist:

	def __init__(self):
		self.NUM_CLASSES = 10;
		# dict to store number of samples per class in training data
		self.SAMPLES_PER_CLASS={}
		path = 'http://yann.lecun.com/exdb/mnist/';
		self.training_labels,self.training_data  = self.loadbatch(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz');
		self.test_labels,self.test_data = self.loadbatch(path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz');
		for i in range(0,10):
			self.SAMPLES_PER_CLASS[str(i)] = len(np.where(self.training_labels == i)[0]);
			#print("Samples for label %s: %i" %(i,self.SAMPLES_PER_CLASS[str(i)]))

	def load_training_batch(self):
		return self.training_data,self.training_labels

	def load_test_batch(self):
		return self.test_data,self.test_labels

	#download data
	def downloadfiles(self,url, force_download=True): 
	    fname = url.split("/")[-1]
	    if force_download or not os.path.exists(fname):
	        if(python_version == 3):
	            urlretrieve(url, fname)
	        else:
	            urllib.urlretrieve(url, fname)
	    return fname

	def loadbatch(self,label_url, image_url):
	    
	    #read label files
	    with gzip.open(self.downloadfiles(label_url)) as labelfile:
	        #Interpret strings as packed binary data. unpack the string to unsigned int of 2x4
	        magic, num = struct.unpack(">II", labelfile.read(8))
	        label = np.fromstring(labelfile.read(), dtype=np.int8)
	   		
	    #read image files
	    with gzip.open(self.downloadfiles(image_url), 'rb') as imagefile:
	        # unpack string to 4x4
	        magic, num, rows, cols = struct.unpack(">IIII", imagefile.read(16))
	        image = np.fromstring(imagefile.read(), dtype=np.uint8)#.reshape(len(label), rows, cols)
	        #image = np.reshape(image,(len(image)*28*28,1)); # store samples in one long array
	        image = image.astype('float32');
	    return (label, image)

	# function that radnomly removes a specified fraction of data samples corresponding to class_id
	def remove_samples(self,data,labels,label_to_remove,fraction):

		# update number of samples
		num_samples = self.SAMPLES_PER_CLASS[str(label_to_remove)]
		samples_to_remove = int(fraction*num_samples)
		self.SAMPLES_PER_CLASS[str(label_to_remove)] = num_samples-samples_to_remove

		total_samples = 0;
		for n in self.SAMPLES_PER_CLASS:
			total_samples += self.SAMPLES_PER_CLASS[n]

		new_data = np.zeros((total_samples*28*28));
		new_labels = np.zeros((total_samples,1));

		# find indices of the elements to remove
		indices = np.where(labels == label_to_remove)[0]
		indices = indices.tolist()
		indices = random.sample(indices,samples_to_remove)
		indices = np.sort(indices)

		indices_index = 0; #index for the indices list
		new_index = 0;
		for i in range(0,len(labels)):
			if( indices_index < len(indices) and indices[indices_index] == i and samples_to_remove > 0):
				samples_to_remove = samples_to_remove-1
				indices_index += 1;			
			else:
				new_labels[new_index] = labels[i];
				new_data[new_index*28*28:(new_index+1)*28*28] = data[i*28*28:(i+1)*28*28]
				new_index+=1;

		return new_data,new_labels

	# removes samples from classes to have the same amount of samples for all classes
	def balance_data(self,data,labels):
		least_samples = self.SAMPLES_PER_CLASS[str(0)];
		for n in self.SAMPLES_PER_CLASS:
			if(self.SAMPLES_PER_CLASS[n] < least_samples):
				least_samples = self.SAMPLES_PER_CLASS[n];
		
		for n in self.SAMPLES_PER_CLASS:
			samples_to_remove = self.SAMPLES_PER_CLASS[n] - least_samples;
			if(samples_to_remove > 0):
				fraction = samples_to_remove/self.SAMPLES_PER_CLASS[n]; # should perhaps use exact number of samples instead
				data,labels = self.remove_samples(data,labels,int(n),fraction) # not super efficient passing data and labels back and forth, but works

		return data,labels

	# JUST FOR VISUALIZATION
	def display_data(self,data,images_per_row):
		img = Image.new('L',(28*images_per_row,28*images_per_row))
		for i in range(0,images_per_row):
			for j in range(0,images_per_row):
				im = data[((i*images_per_row)+j)*784:((i*images_per_row)+j+1)*784];
				im = np.reshape(im,(28,28))
				im = Image.fromarray(im)
				img.paste(im,(j*28,i*28))
		img.show()

	# Augments data so that all classes have the same amount of samples corresponding to the maximum 
	# possible modes: oversample - duplicate existing sample, random - use any augmentaiton method for new sample
	def augment_data(self,data,labels,mode='oversample'):
		print("Augmentation mode:", mode)
		max_samples = 0;
		for n in self.SAMPLES_PER_CLASS:
			if(self.SAMPLES_PER_CLASS[n] > max_samples):
				max_samples = self.SAMPLES_PER_CLASS[n];

		data = np.reshape(data,(-1,28,28));
		data_length = len(data);
		for n in self.SAMPLES_PER_CLASS:
			samples_to_create = max_samples - self.SAMPLES_PER_CLASS[n];
			data_length += samples_to_create;

		labels = np.reshape(labels,((-1)))

		output_labels = np.zeros((data_length));
		output_data = np.zeros((data_length,28,28));
		output_labels[0:len(labels)] = labels[:];
		output_data[0:len(data)] = data[:];

		start_index = len(labels);
		for n in self.SAMPLES_PER_CLASS:
			samples_to_create = max_samples - self.SAMPLES_PER_CLASS[n];
			if(samples_to_create > 0):
				print("Samples to create for label %s: %i" %(n,samples_to_create));
				# augment the data
				new_samples_data,new_samples_labels = self.create_samples(data,labels,int(n),samples_to_create,mode);
				#add to data and add corresponding labels to labels
				output_labels[start_index:start_index+samples_to_create] = new_samples_labels[:];
				output_data[start_index:start_index+samples_to_create] = new_samples_data[:];
				start_index += samples_to_create;

		p = np.random.permutation(len(output_labels));
		output_labels = output_labels[p];
		output_data = output_data[p];
		output_data = np.reshape(output_data,(len(output_data)*28*28));
		return output_data,output_labels;

	def create_samples(self,data,labels,label,number_of_samples,mode='oversample'):
		old_data = np.reshape(data,(-1,28,28))
		new_samples_data = np.zeros((number_of_samples,28,28))
		new_samples_labels = np.zeros((number_of_samples));
		indices = np.where(labels == label)[0];
		
		for i in range(0,number_of_samples):
			if(mode == 'oversample'):
				random_index = np.random.choice(indices);
				new_samples_labels[i] = labels[random_index];
				new_samples_data[i] = old_data[random_index];
			elif(mode == 'random'):
				random_index = np.random.choice(indices); #pick random sample to use as basis for new sample
				new_samples_data[i],new_samples_labels[i] = self.create_transformed_sample(old_data[random_index],labels[random_index])
			else:
				raise Exception("Invalid data augmentation mode picked")

		new_samples_data = new_samples_data.astype(np.float32);
		return new_samples_data,new_samples_labels;

	def create_transformed_sample(self,sample,label):
		random_rotation = 20*np.random.rand()-10 	# interval [-10,10]
		random_translation_x = 4*np.random.rand()-2 # interval [-2,2] is probably max for mnist
		random_translation_y = 4*np.random.rand()-2 # interval [-2,2]
		tform = tf.SimilarityTransform(scale=1,rotation=0,translation=(random_translation_x,random_translation_y)) 
		sample = sample.astype(np.uint8) # need to use uint8 for transformations?
		transformed_sample = tf.warp(sample,tform)
		transformed_sample = tf.rotate(transformed_sample,random_rotation); # returns matrix of same shape but values between 0-1
		transformed_sample = transformed_sample*255
		return transformed_sample,label

	def print_sample_distribution(self,labels):
		SAMPLES_PER_LABEL = {};
		for i in range (0,10):
			SAMPLES_PER_LABEL[str(i)] = len(np.where(labels == i)[0]);
			print("Samples for label %s: %i" %(i,SAMPLES_PER_LABEL[str(i)]))

	# returns number of samples for class corresponding to class_label
	def count_samples_for_class(self,labels,class_label):
		return len(np.where(labels == class_label)[0]);

	# Transforms an array of labels into a one-hot representation
	def make_one_hot(self,labels):
		labels = np.reshape(labels,(-1)).astype(int)
		binarizer = sklearn.preprocessing.LabelBinarizer();
		binarizer.fit(range(max(labels)+1));
		one_hot_labels = binarizer.transform(labels); 
		return one_hot_labels;
	    
	# return tuple with two ndarrays 
	def get_next_batch(self,training_data,training_labels,batch_size):
		data = np.reshape(training_data,(-1,200))
		

		p = np.random.permutation(len(training_labels));
		training_labels = training_labels[p];
		data = data[p];

		x = data[0:batch_size]
		y = training_labels[0:batch_size]
		return (x,y)

'''
#tests the class
mnist = Mnist();
training_data,training_labels = mnist.load_training_batch()
training_labels = mnist.make_one_hot(training_labels)

batch = mnist.get_next_batch(training_data,training_labels,5);
mnist.display_data(np.reshape(batch[0],(-1)),5)
batch = mnist.get_next_batch(training_data,training_labels,5);
print(batch[1][0:5,:])
mnist.display_data(np.reshape(batch[0],(-1)),5)

test_data,test_labels = mnist.load_test_batch()
test_data = np.reshape(test_data,(-1,784))
test_labels = mnist.make_one_hot(test_labels)
'''

#new_data,new_labels = mnist.augment_data(training_data,training_labels,'random');
#mnist.display_data(new_data,35)

