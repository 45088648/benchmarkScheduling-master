# Training and Testing RBFNN

# Import labs
import numpy as np
import matplotlib.pyplot as plt
from RBFN import RBFN
import sys
import numpy as np
import FileProcess as fipr
import time

# Start counting time
start_time = time.clock()

# Open and load csv files
time_load_start = time.clock()
X_train, y_train = fipr.load_csv("train_file.csv", True)
X_test, y_test = fipr.load_csv("test_file.csv", True)
y_train = y_train.flatten() 
y_test = y_test.flatten()
time_load_end = time.clock()
print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

# Training the network
'''
x = np.linspace(0,10,100)
y = np.sin(x)
'''
# start counting time for training
time_train_start = time.clock()

# start training
model = RBFN(input_shape = 1, hidden_shape = 20)
model.fit(X_train,y_train)

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

# Calculate running time
print("--- Total running time: %g seconds ---" % (time.clock() - start_time))

'''
plt.plot(x,y,'b-',label='real')
plt.plot(x,y_pred,'r-',label='fit')
plt.legend(loc='upper right')
plt.title('Interpolation using a RBFN')
plt.show()
'''
