### Back-Propagation Neural Networks

# Import libraries
import sys
import numpy as np
import string
import FileProcess as fipr
import time

def rand(a,b):
    return (b-a)*np.random.random_sample() + a

def sigmoid(x):
    # Symmetrical sigmoid
    return np.tanh(x)
    # return 1./(1.+np.exp(-x)) # normal sigmoid, range 0 to 1

vsigmoid = np.vectorize(sigmoid)

def dsigmoid(y):
    return 1.0 - y**2

vdsigmoid = np.vectorize(dsigmoid)

class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # Initialize numpy arrays of ones with default dtype float
        self.ai = np.ones((self.ni,1), dtype=float)
        self.ah = np.ones((self.nh,1), dtype=float)
        self.ao = np.ones((self.no,1), dtype=float)

        # initialize weights
        # Make random matrix with values in range [-0.2, 0.2)
        self.wi = (np.random.random_sample((self.ni, self.nh)) - 0.5) * 0.4
        # Make random matrix with values in range [-2., 2.)
        self.wo = (np.random.random_sample((self.nh, self.no)) - 0.5) * 4.
                
        # last change in weights for momentum
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError("Wrong number of inputs")

        # input activations
        for i in range(self.ni-1):
            self.ai[i,0] = inputs[i]

        # hidden activations
        # shapes: (nh,1) = (nh,sni) x (sni,1)
        self.ah = vsigmoid( self.wi.T.dot(self.ai) )

        # output activations
        # shapes: (no,1) = (no,nh) x (nh,1)
        self.ao = vsigmoid( self.wo.T.dot(self.ah) )

        return self.ao

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('Wrong number of target values')

        # Assume 'targets' has shape (no,1)
        error = targets - self.ao
        # vectorized sigmoid followed by element-wise multiplication with errors
        # (no,1) * constant:
        output_deltas = vdsigmoid(self.ao) * error

        # (nh,1) = (nh,no) matrix times (no,1) column vector:
        error = self.wo.dot(output_deltas)
        # (nh,1) * constant:
        hidden_deltas = vdsigmoid(self.ah) * error

        # (nh,no) matrix = (nh,1) column vector times (1,no) row vector:
        change = self.ah.dot(output_deltas.T)
        self.wo += N*change + M*self.co
        self.co = change

        # (self.ni,nh) = (self.ni,1) x (1,nh):
        change = self.ai.dot(hidden_deltas.T)
        # Each of these objects (except N,M) has shape (self.ni,nh):
        self.wi += N*change + M*self.ci
        # (self.ni,nh):
        self.ci = change

        # Vector subtraction, element-wise exponentiation, then sum over [self.no] elements 
        error = np.sum(0.5*(targets - self.ao)**2)
        return error

    def test(self, patterns):
        c = 0
        for p in patterns:
            a = p[1]
            b = self.update(p[0])
            
            #print(p[1], '->', self.update(p[0]))
            if abs(a-b) < 0.06:
                c+=1
        accuracy = c/len(patterns)
        print("Accuracy:  %g" % accuracy)
            

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])

        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        errs = []
        for i in range(iterations):
            error = 0.0
            # Do backpropagation training one data point at time:
            for p in patterns:
                inputs = p[0]
                if type(p[1]) == list:
                    targets = p[1]
                else: targets = list([p[1]])
                self.update(inputs)
                error = error + self.backPropagate(np.array(targets).reshape((len(targets),1)), N, M) 
            errs.append(error)
            if i % 100 == 0:
                print('error %-0.5f' % error)
        return errs
    # Uncomment this to get training error as function of iteration
    
def normalize(data):
    # A slightly hacky normalization ncode
    colmin = np.min(data, axis=0)
    colmax = np.max(data, axis=0)
    colmean = np.mean(data, axis=0)
    colrange = colmax-colmin+0.001 # the hacky part
    #print(colmin,colmax,colmean,colrange)
    data_norm = data - colmean
    # if colrange > 0.001:
    data_norm = data_norm / colrange
    return data_norm

# for test using iris data
def irisdemo():
    from sklearn import datasets
    iris = datasets.load_iris()
    pattern = []

    data_norm = normalize(iris.data)
    # Classify 3 different iris types with two tanh output nodes:
    # Encode class 0 as [0,0]
    # Encode class 1 as [1,0]
    # Encode class 2 as [0,1]
    encode = {0:[0.,0.], 1:[1.,0.], 2:[0.,1.]}

    for i,x in enumerate(data_norm):
        pattern.append([ x, encode[iris.target[i]] ])
    print(pattern)

    n = NN(4,9,2)
    n.train(pattern,iterations=1000,N=0.03,M=0.06)
    n.test(pattern)

# load data, create NN, train and test
def main():
    
    # open and load csv files
    time_load_start = time.clock()
    X_train, y_train = fipr.load_csv("train_file.csv", True)
    X_test, y_test = fipr.load_csv("test_file.csv", True)
    y_train = y_train.flatten() 
    y_test = y_test.flatten()
    time_load_end = time.clock()
    print("Loading finished, loading time: %g seconds" % (time_load_end - time_load_start))

    X_test_even, y_test_even = fipr.load_csv("test_file_even.csv", True)
    y_test_even = y_test_even.flatten()
    # scale features to encourage gradient descent convergence
    X_train = fipr.scale_features(X_train, 0.0, 1.0)
    X_test = fipr.scale_features(X_test, 0.0, 1.0)
    
    X_test_even = fipr.scale_features(X_test_even, 0.0, 1.0)
    
    
    Pattern_train = []
    for i,sample_train in enumerate(X_train):
        Pattern_train.append([ sample_train, y_train[i] ])
        
    Pattern_test = []
    for j,sample_test in enumerate(X_test):
        Pattern_test.append([ sample_test, y_test[j] ])

    Pattern_test_even = []
    for k,sample_test_even in enumerate(X_test_even):
        Pattern_test_even.append([ sample_test_even, y_test_even[k] ])
        
    #print(Pattern_train)
    #print(Pattern_test)
    # Teach network XOR function (for test only)
    '''pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
        ]
    print(pat)

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)'''
    
    # Test on Iris data
    #pattern = irisdemo()
    
    # create a network with two hundred inputs, two hidden, and one output nodes
    n = NN(200, 4, 1)

    # start counting time for training
    time_train_start = time.clock()
    
    # train it with some patterns
    n.train(Pattern_train)

    # print training time
    time_train_end = time.clock()
    print("Training finished, training time: %g seconds \n" % (time_train_end - time_train_start))

    # start counting time for testing
    time_test_start = time.clock()
    
    # test it
    n.test(Pattern_test)

    # print testing time
    time_test_end = time.clock()
    print("Testing finished, testing time: %g seconds  \n" % (time_test_end - time_test_start))

    # test on EVEN data set
    n.test(Pattern_test_even)


if __name__ == '__main__':
	"""
	The main function is called when BP Neural Network.py is run from the command line with arguments.
	"""
	start_time = time.clock()
	args = sys.argv[1:] # get arguments from the command line
	main( *args )
	print("--- Total running time: %g seconds ---" % (time.clock() - start_time))
