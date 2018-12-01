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
    mu_true, sigma_true = 0, 0.1    # Define average and variance of norm distribution
    mu_false, sigma_false = 1, 0.1  # Different average and same variance for false data

    X_true = []                     # X is 2d array for samples, Y is 1d array for labels
    X_false =[]
        
    data_set_true = []
    data_set_false = []
    data_set_origin = []
    data_set = []

    sec_num_d3 = 4                  # Number of sub-label in depth 3
    sec_num_d2 = 8                  # Number of sub-label in depth 2
    sec_num_d1 = 40

    singl_rate = 0.05
    singl_num = int(singl_rate*data_size)       # Number of singluar points
    #sec_size = int(attribute_num/sec_num)       # How many (50) attributes creates one sub-label

    y = 0                           # Initialize final label

    X_d3 = np.zeros(sec_num_d3)
    X_d2 = np.zeros(sec_num_d2)
    X_d1 = np.zeros(sec_num_d1)

    # Start creating data with label = 0
    for count in range(0, int(data_size*(1-false_percent))):
        # X_d3 = [0,0,0,0] in this case
        '''print("X_d3 is:")
        print(X_d3)'''
        
        for i in range(0, sec_num_d3):
            temp1 = np.random.choice([0, 1])
            if temp1 == 0:
                temp2 = np.random.choice([0, 1])
            else: temp2 = 0

            X_d2[2*i] = temp1
            X_d2[2*i+1] = temp2         # Now we have X_d2 as array with lenth 8

        '''print("X_d2 is:")
        print(X_d2)'''

        for i in range(0, sec_num_d2):
            if X_d2[i] == 0:
                sum_xd1 = np.random.choice([0, 1, 2])
            else: sum_xd1 = np.random.choice([3, 4, 5])

            temp_xd1 = np.zeros(5)
        
            for j in range(1, sum_xd1+1):
                temp_xd1[5-j] = 1

            if sum_xd1 == 0:
                temp_xd1 = np.zeros(5)

            for k in range(0, 5):
                X_d1[i*5+k] = temp_xd1[k]        # Now we have X_d1 as array with lenth 40

        x = np.zeros(attribute_num)
    
        for i in range(0, sec_num_d1):
            if X_d1[i] == 0:
                temp_x = np.random.normal(mu_true, sigma_true, 5)
                for j in range(0, 5):
                    x[i*5+j] = temp_x[j]

            if X_d1[i] == 1:
                temp_x = np.random.normal(mu_false, sigma_false, 5)
                for j in range(0, 5):
                    x[i*5+j] = temp_x[j]

        x = np.asarray(x)
        #print(x)
        x = np.append(x, y)
        #print(x)

        X_true = np.append(X_true, x, axis=0)
        # X_true is the data matrix with size [data_size, attribute+1], last element is label

    X_true = np.reshape(X_true, (int(data_size*(1-false_percent)), attribute_num+1))

    data_set_true = X_true

    # Start Creating data with label = 1
    y = 0
    sec_false_check = 1
    
    for count in range(0, int(data_size*false_percent)):
        
        while (sec_false_check):                                   # Create X_d3 = [random 0 and 1s]
            X_d3 = np.random.choice([0,1], size = (sec_num_d3,))
            if sum(X_d3) > 0:
                sec_false_check = 0
                y = 1
            else: sec_false_check = 1
            
        '''print("X_d3 is:")
        print(X_d3)'''
        
        for i in range(0, sec_num_d3):
            if X_d3[i] == 0:
                temp1 = np.random.choice([0, 1])
                if temp1 == 0:
                    temp2 = np.random.choice([0, 1])
                else: temp2 = 0

                X_d2[2*i] = temp1
                X_d2[2*i+1] = temp2         

            if X_d3[i] == 1:
                temp1 =1
                temp2 =1
                X_d2[2*i] = temp1
                X_d2[2*i+1] = temp2
                # Now we have X_d2 as array with lenth 8

        '''print("X_d2 is:")
        print(X_d2)'''

        for i in range(0, sec_num_d2):
            if X_d2[i] == 0:
                sum_xd1 = np.random.choice([0, 1, 2])
            else: sum_xd1 = np.random.choice([3, 4, 5])

            temp_xd1 = np.zeros(5)
        
            for j in range(1, sum_xd1+1):
                temp_xd1[5-j] = 1

            if sum_xd1 == 0:
                temp_xd1 = np.zeros(5)

            for k in range(0, 5):
                X_d1[i*5+k] = temp_xd1[k]        # Now we have X_d1 as array with lenth 40

        x = np.zeros(attribute_num)
    
        for i in range(0, sec_num_d1):
            if X_d1[i] == 0:
                temp_x = np.random.normal(mu_true, sigma_true, 5)
                for j in range(0, 5):
                    x[i*5+j] = temp_x[j]

            if X_d1[i] == 1:
                temp_x = np.random.normal(mu_false, sigma_false, 5)
                for j in range(0, 5):
                    x[i*5+j] = temp_x[j]

        x = np.asarray(x)
        x = np.append(x, y)

        X_false = np.append(X_false, x, axis=0)
        # X_false is the data matrix with size [data_size, attribute+1], last element is label

    X_false = np.reshape(X_false, (int(data_size*false_percent), attribute_num+1))
    data_set_false = X_false
    data_set_origin = np.append(data_set_true, data_set_false, axis=0)

    np.random.shuffle(data_set_origin)        
    #print (data_set_origin)
    
    # Add singular points to database by reversing labels
    
    singl_index = np.random.choice(data_size, singl_num)
    for s in range (0, singl_num):
        temp = data_set_origin[singl_index[s]]
        temp[attribute_num] = abs(temp[attribute_num]-1)
        data_set_origin[singl_index[s], attribute_num] = temp[attribute_num]
    
                
        
    return data_set_origin
        

def main():
    # produce training dataset
        
    train_data = creat_data(100000, 200, 0.1)
    np.savetxt("train_file_deep.csv", train_data, delimiter=",")
        
    
    # produce test dataset
    test_data = creat_data(50000, 200, 0.1)
    np.savetxt("test_file_deep.csv", test_data, delimiter=",")
    
    # produce test dataset
    test_data_even = creat_data(50000, 200, 0.5)
    np.savetxt("test_file_deep_even.csv", test_data_even, delimiter=",")                       



if __name__ == '__main__':
    """
    The main function is called when logistic.py is run from the command line with arguments.
    """
    args = sys.argv[1:] # get arguments from the command line
    main( *args )


     
    
