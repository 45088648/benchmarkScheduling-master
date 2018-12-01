### Hypothetical Data Production for Shop-floor (Improved)###

# Import libraries
import numpy as np
import os
import sys
import urllib
import math
import tensorflow as tf


# Define data metrics

'''
train_size = 100000
eval_size  = 50000
attribute_num = 200

false_percent = 0.1 or 0.5
'''
# Define patterns: true and false
def creat_data(data_size, attribute_num, false_percent):

        #polynominal y = ax_n^n + ax_(n-1)^(n-1) + ... +b !!!This is old pattern, not used!!!
        #New Pattern: x1-x50 => y1  x51-x100 => y2 ... y3, y4
        #Final label y = y1 or y2 or y3 or y4   if there is one yi = 1 then y = 1
        a = 1
        b = 0
        b_false = 1
        mu_true, sigma_true = 0, 0.1    # Define average and variance of norm distribution
        mu_false, sigma_false = 1, 0.1  # Different average and same variance for false data
        X_true = []                     # X is 2d array for samples, Y is 1d array for labels
        
        X_false =[]
        
        data_set_true = []
        data_set_false = []
        data_set_origin = []
        data_set = []

        sec_num = 4                         # Number of sub-label in data
        #print(sec_num)
        singl_rate = 0.05
        singl_num = int(singl_rate*data_size)        # Number of singluar points
        sec_size = int(attribute_num/sec_num)    # How many (50) attributes creates one sub-label
        y_sec = np.zeros(sec_num)
        #print(y_sec)
        y = 0                               # Initialize final label

        for i in range(0, int(data_size*(1-false_percent))):
                
                x = np.random.normal(mu_true, sigma_true, attribute_num)
                x = np.asarray(x)
                
                for k in range (0, sec_num):
                        
                        for j in range (k*sec_size, (k+1)*sec_size):
                            y_sec[k] +=  x[j]
                            
                        y_sec[k] = y_sec[k]/sec_size

                        if y_sec[k]<0.5 and y_sec[k]>-0.5:
                                y_sec[k] = 0
                        else: y_sec[k] = 1
      
                y = y_sec[0] or y_sec[1] or y_sec[2] or y_sec[3] 
                x = np.append(x, y)
                X_true = np.append(X_true, x, axis=0)
                # X_true is the data matrix with size [data_size, attribute+1], last element is label

        
        
        X_true = np.reshape(X_true, (int(data_size*(1-false_percent)), attribute_num+1))
        #print(X_true)
        
        data_set_true = X_true    
        #print(data_set_true)

        
        y=0
        sec_faulse_check = 1
        
        for i in range(0,int(data_size*false_percent)):
                
                x = np.random.normal(mu_true, sigma_true, attribute_num)
                x = np.asarray(x)

                while (sec_faulse_check):                               # Create y_sec = [random 0 and 1s]
                        y_sec = np.random.choice([0,1], size = (sec_num,))
                        if sum(y_sec) > 0:
                            sec_faulse_check = 0
                            y = 1
                        else: sec_faulse_check = 1
            
                for k in range (0, sec_num):
                        if y_sec[k] == 1:
                                for m in range (k*sec_size, (k+1)*sec_size):
                                        x[m] = np.random.normal(mu_false, sigma_false)
                                        '''y_sec[k] = y_sec[k] + x[m]

                                y_sec[k] = y_sec[k]/sec_size
                        
                                if y_sec[k]>0.5 and y_sec[k]<1.5:
                                        y_sec[k] = 1
                                else: y_sec[k] = 0'''
                
                y = y_sec[0] or y_sec[1] or y_sec[2] or y_sec[3]
                x = np.append(x, y)
                X_false = np.append(X_false, x, axis=0)
                

        X_false = np.reshape(X_false, (int(data_size*false_percent), attribute_num+1))
        #print(X_false)
        data_set_false = X_false
        data_set_origin = np.append(data_set_true, data_set_false, axis=0)
        #print (data_set_origin)
        
        #Shuffle the training set to mix true and false samples together
        np.random.shuffle(data_set_origin)        
        print (data_set_origin)

        # Add singular points to database by reversing labels
        #'''
        singl_index = np.random.choice(data_size, singl_num)
        for s in range (0, singl_num):
                temp = data_set_origin[singl_index[s]]
                temp[attribute_num] = abs(temp[attribute_num]-1)
                data_set_origin[singl_index[s], attribute_num] = temp[attribute_num]
        #'''
                
        
        return data_set_origin


# Create singularity for robustness test
#def singular_points():
# method 1: create true data but with false label
# method 2: create data with very different distribution and unknown label
                       
                   
def main():
        # produce training dataset
        
        train_data = creat_data(100000, 200, 0.1)
        np.savetxt("train_file.csv", train_data, delimiter=",")
        
       
        # produce test dataset
        test_data = creat_data(50000, 200, 0.1)
        np.savetxt("test_file.csv", test_data, delimiter=",")
                       



if __name__ == '__main__':
	"""
	The main function is called when logistic.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )
