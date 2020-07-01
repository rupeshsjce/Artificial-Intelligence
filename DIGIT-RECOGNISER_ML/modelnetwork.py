from six.moves import range
import random
import numpy as np
import time
import sys
import os.path
from os import path


class Categotical_cross_entropy:
    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels, epsilon=0.00001):
        return -np.sum(labels * np.log(epsilon+predictions))
   
    @staticmethod 
    def loss_derivative(p,y): 

        epsilon = 0.0001
        assert(p.shape == y.shape and p.shape[1] == 1)
        py = p[y == 1]
        assert(py.size == 1)
        D = np.zeros_like(p)
        D[y == 1] = -1/(epsilon + py.flat[0]) # D is (10,1)
        D1 = D.flatten() # D1 is rank array (10,)
        return D
       
    
class Sequential:
    def __init__(self):
        print("Init Sequential ...")
        self.layers = []
        self.test_loss = 0
        self.global_params = []
        self.loss = Categotical_cross_entropy()

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].join(self.layers[-2])

    def fit(self, training_data, num_iter, learning_rate,
              mini_batch_size, test_data=None): 
        n = len(training_data)
        not_allowed = False
        if n >=10000:
            self.init_wb(not_allowed)
            
        # num_iter may be less but let's calculate train time and then train
        
        fit_start_time = time.time()
        max_time = 1500 # 1500s and remaining 300s=6min for other stuff (150s is enough for rest)
        for epoch in range(num_iter):  
            start_time = time.time()
            if not (time.time() - fit_start_time < max_time):
                print("Ending the Training as time reached the max limit :", time.time() - fit_start_time)
                break
            #Shuffle the training data randomly so that we don't train same pattern of data everytime
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for
                k in range(0, n, mini_batch_size)
            ]
            # Trainig data
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch, learning_rate)
            # Testing data    
            if test_data:
                n_test = len(test_data)
                test_result = self.evaluate(test_data)
                accuracy = test_result/n_test
                print("Epoch {0}/{1}: {2} / {3} Test-Loss: {4} Test-Accuracy: {5}"
                      .format(epoch, num_iter, test_result, n_test, self.test_loss/n_test, accuracy))  
                self.test_loss = 0
            else:
                print("Epoch {0} complete".format(epoch))
            print("Execution time: ", epoch, time.time() - start_time)    


    def train_mini_batch(self, mini_batch, learning_rate):
        self.forwardFeed_backpropagation(mini_batch)
        self.update_layer_params(mini_batch, learning_rate)


    def update_layer_params(self, mini_batch, learning_rate):
        # we are dividing because delta_weight and delta_bias for any DenseLayer is summation 
        # over all the examples in mini_bacth. So while fiting the model, send proper learning rate. 
        learning_rate = learning_rate / len(mini_batch) 
        for layer in self.layers:
            layer.update_weights_bias(learning_rate) 
        for layer in self.layers:
            layer.reset_deltas()  


    def forwardFeed_backpropagation(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].data_in = x
            for layer in self.layers:
                layer.forwardFeed() 
            # This will be (10, 1) matrix  or <numpy.ndarray>  
            self.layers[-1].delta_in = \
                self.loss.loss_derivative(self.layers[-1].data_out, y)
            for layer in reversed(self.layers):
                layer.backpropagation(y) 
        
            
    def init_wb(self, not_allowed):
        if not not_allowed:
            if path.exists("hidden_params.npy"):
                pass
            else:
                return None
            
            self.global_params = np.load('hidden_params.npy', allow_pickle= True)
            i, j = 0, 0
            
            for layer in self.layers:
                if (i % 2 == 0):
                   layer.params =  self.global_params[j]
                   layer.weight =  layer.params[0]
                   layer.bias   =  layer.params[1]
                   #print(layer.params)
                   j = j + 1
                   
                i = i + 1
        
        
    def forward_feed_for1digit(self, x, y): 
        self.layers[0].data_in = x
        for layer in self.layers:
                layer.forwardFeed()

        V =  np.argmax(self.layers[-1].data_out) 
        self.test_predictions.append([V]) # V is only needed if using dataframe

        predictions = self.layers[-1].data_out
        self.test_loss += self.loss.loss_function(predictions, y) 
        return predictions

    def evaluate(self, test_data): 
        self.test_predictions = []
        test_results = [(
            np.argmax(self.forward_feed_for1digit(x,y)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

