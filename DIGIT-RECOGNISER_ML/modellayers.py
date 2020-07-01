import numpy as np
import sys


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


def ReLU_function( x, derivative=False ):
    if x.all() == 0:
        print("ERRRRRRR as x has 0", x)
    if derivative:
        retVal =  (x > 0).astype(float)
    else:
        # Return the activation signal
        retVal = np.maximum( 0, x )
    return retVal    


def LReLU_function( x, derivative=False, leakage = 0.01 ):
    if derivative:
        # Return the partial derivation of the activation function
        return np.clip(x > 0, leakage, 1.0)
    else:
        # Return the activation signal
        output = np.copy( x )
        output[ output < 0 ] *= leakage
        return output


#stableSoftmax
def softmax(X):
    
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


#Partial derivatives of softmax function.
def dsoftmax(z):
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D # (10,10) matrix

def sigmoid_helper(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid(z):
    return np.vectorize(sigmoid_helper)(z)

def dsigmoid_helper(x):
    return sigmoid_helper(x) * (1 - sigmoid_helper(x))

#Partial derivatives of sigmoid function.
def dsigmoid(z):
    D = np.vectorize(dsigmoid_helper)(z)
    return D


class Layer(object):
    def __init__(self):
        self.previous_val = 0
        self.next_val = 0
        self.params = []
        self.previous = None
        self.next = None 
        self.data_in = None
        self.data_out = None
        self.delta_in = None 
        self.delta_out = None
        self.verbose = False

    def join(self, layer): 
        self.previous = layer
        layer.next = self

    def get_previous_outputData(self):  
        if self.previous is not None:
            return self.previous.data_out
        else:
            return self.data_in

    def get_next_outputDelta(self): 
        if self.next is not None:
            return self.next.delta_out
        else:
            return self.delta_in



class ActivationLayer(Layer):
    def __init__(self, input_dim, name, activation):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.name = name
        self.activation = activation
        print("Activation function : ", self.name, self.activation)

    def forwardFeed(self):
        data = self.get_previous_outputData()
    
        if self.activation == "sigmoid":
           self.data_out = sigmoid(data)
        elif self.activation == "ReLU":   
           #print("Going to call ReLU only from ", self.name)
           self.data_out = ReLU_function(data, False) # ReLU
        elif self.activation == "LReLU":   
           self.data_out = LReLU_function(data, False) # Leaky ReLU
         
        #Use SOFTMAX for Activation layer used for output layer
        elif self.activation == "softmax": 
           self.data_out = softmax(data)
        else:
           print("Invalid activation layer for layer : ", self.name)
            
        #print("self.data_out shape :", self.data_out.shape)

    def backpropagation(self,y):
        delta = self.get_next_outputDelta() # (n(l) , 1)
        data = self.get_previous_outputData()   # (n(l) , 1)
  
        # d(J)/d(zA2) or d(J)/d(aA2) , d(J)/d(zA1)
        if self.activation == "sigmoid":
           d = dsigmoid(data)  # (n(l), 1)
           self.delta_out = delta * d
        elif self.activation == "ReLU":   
           #print("Going to call ReLU derivative from ", self.name)
           d = ReLU_function(data, True) # ReLU derivative
           self.delta_out = delta * d
        elif self.activation == "LReLU":   
           d = LReLU_function(data, False) # Leaky ReLU derivative
           self.delta_out = delta * d
           #print("LReLU derivate shape and delta shape should match, ", d.shape, delta.shape)

        elif self.activation == "softmax":
           #print("Calling from here ...")
           # d(J)/d(zA3) or d(J)/d(aA3)
           self.delta_out = delta * dsoftmax(data) # (10,10) matrix
            
           y1= np.argmax(y) # index=2 of digit where it is 1 [0,0,1,0,0,0,0,0,0,0]
           self.delta_out = self.delta_out[y1]
           self.delta_out = self.delta_out.reshape(10,1) # finally (10 * 1) matrix
        else:
            print("[D] Invalid activation layer for layer : ", self.name)


    def save_weight_bias(self):
        pass
    
    def load_weight_bias(self):
        pass

    def update_weights_bias(self, learning_rate):
        pass
    
    def reset_deltas(self):
        pass    
    
    

class DenseLayer(Layer):

    def __init__(self, input_dim, output_dim, name, w_init): 

        super(DenseLayer, self).__init__()

        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_init = w_init
        
        
          
        if self.w_init =="XavierUniform":
            print("XavierUniform : ", self.name)
            #Xavier/Glorot Uniform Initialization.
            nin = input_dim
            nout = output_dim
            sd = np.sqrt(6.0 / (nin + nout))
            
            #self.weight = np.random.randn(output_dim, input_dim)
            self.weight = np.random.uniform(-sd, sd, size=(output_dim,input_dim))
            self.weight = np.array(self.weight, dtype=np.float128)
            
            
            nin = 1
            nout = output_dim
            sd = np.sqrt(6.0 / (nin + nout))
            #self.bias = np.random.randn(output_dim, 1)
            self.bias = np.random.uniform(-sd, sd, size=(nout,nin))
            self.bias = np.array(self.bias, dtype=np.float128)
        
        elif self.w_init =="HeNormal":
            print("HeNormal : ", self.name)
        
            self.weight = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
            self.weight = np.array(self.weight, dtype=np.float128)
            self.bias = np.random.randn(output_dim, 1) * np.sqrt(2/input_dim)
            self.bias = np.array(self.bias, dtype=np.float128)

        else:
            print("Invalid weight initilization for layer :", self.name)
            
            
        self.params = [self.weight, self.bias] 

        self.delta_weight = np.zeros(self.weight.shape)  
        self.delta_bias = np.zeros(self.bias.shape)


    def save_weight_bias(self):
        pass
        
        
    def load_weight_bias(self):
        pass



    def forwardFeed(self):
        data = self.get_previous_outputData()
        self.data_out = np.dot(self.weight, data) + self.bias 


    def backpropagation(self,y):
    
        #(n(l+1) * 1) 
        #Analogy where hidden and activation is same node. dz(l) = da[l] * g(l)' * z[l] = w[l+1].transpose * dz[l+1] * g(l)' * z[l] 
        delta = self.get_next_outputDelta() 

        data = self.get_previous_outputData() #(n(l-1), 1) or a[l-1]
        
        # d(J)/d(wOL) , d(J)/d(wH2), d(J)/d(wH1) and so for base 
        self.delta_bias += delta  # <2> (n(l+1), 1)
        self.delta_weight += np.dot(delta, data.transpose())  #  (n(l+1), n(l-1))
        
        #The backward pass is completed by passing an output delta to the previous layer.
        # zOL means OL.data_out
        # d(J)/d(zOL), d(J)/d(zH2), d(J)/d(zH1)
        self.delta_out = np.dot(self.weight.transpose(), delta)  #  (n(l-1), 1)


    def update_weights_bias(self, rate): 
        self.weight -= rate * self.delta_weight
        self.bias -= rate * self.delta_bias

    def reset_deltas(self):  
        self.delta_weight = np.zeros(self.weight.shape)
        self.delta_bias = np.zeros(self.bias.shape)

