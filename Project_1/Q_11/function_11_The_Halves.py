import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as q
import sklearn
from sklearn.model_selection import GridSearchCV
import time

def shuffle_N_split(df, position_label, size_train):
    np.random.seed(1804475) # Mina's matricola number
    rang = np.arange(df.shape[0]) # list from 0 to 249
    np.random.shuffle(rang) # shuffle the data list
    train = df[rang[0:size_train]] # takels the first 186 elements
    test = df[rang[size_train:]] # take the remaining
    x_train = train[:, :position_label]
    y_train = train[:, position_label]
    x_test = test[:, :position_label]
    y_test = test[:, position_label]
    return x_train, y_train, x_test, y_test

def train(x_train, y_train, N, n, m, rho, seed):
        
    # generate W and V
    W = input_hidden_weights(N, n, seed, 0, 1)
    V = hidden_output_weights(m, N, seed, 0, 1)

    # union in the same vector W and V
    omega = np.vstack([W.reshape(-1,1), V.reshape(-1,1)])
    
    args = [W.shape[0], W.shape[1], V.shape[0], V.shape[1], x_train, rho, y_train.reshape(-1,1) , N, True]
    
    # find best parameter
    start_time = time.time()
    optimizer = sc.optimize.minimize(loss, omega, args=args, method="BFGS", jac=gradient, tol=0.0001, options={"maxiter":3000})
    end = time.time() - start_time 
    optimi = optimizer['x']
    
    # recalculate W and V
    W_new = optimi[: W.shape[0] * W.shape[1]].reshape(W.shape[0], W.shape[1])
    V_new = optimi[W.shape[0] * W.shape[1]: ].reshape(V.shape[0], V.shape[1])

    # Final Train Error
    y_train_pred = feedforward(x_train, W_new, V_new)
    
    return optimizer, y_train_pred, end, W_new, V_new, args


# N: number of neurons in the first hidden layer
# n: the dimension of input, in this project is 2
def hidden_output_weights(m, N, seed, start, end):
    np.random.seed(seed)
    # In this case m=1 and we don't consider biases for output layer
    return np.random.uniform(start, end,size=(m, N))             
  
# m : the dimension of output
def input_hidden_weights(N, n, seed, start, end):
    np.random.seed(seed)
     # We added a default +1 that corresponds to the bias
    return np.random.uniform(start, end,size=(N, n + 1))        

def g(t):
    return np.tanh(t)

def feedforward(x, W, V):
    t = np.dot(W,x)
    layer1 = g(t)
    output = np.dot(V, layer1)
    return output

def loss(omega, args):
    row_w, col_w, row_v, col_v, x, rho, y, N, reg = args 
    norma = np.sum(omega ** 2)
    t = np.dot(omega[: row_w * col_w ].reshape(row_w, col_w), x) # (200, 186) W @ x         
    temp = g(t) 
    pred = np.dot(omega[row_w * col_w:].reshape(row_v, col_v), temp).reshape(-1,1)      # V @ t 
    p = len(y)
    regularization = (rho / 2) * norma 
    t1 = 1/(2*p)
    error = np.sum((pred - y)**2)  #((186, 1), (186, 1))
    return t1 * error  + regularization * reg


def plotting(funct, title, W, V, d ):
    #create the object empty figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #create the grid

    x = np.linspace(-2, 2, d) #create 50 points between [-5,5] evenly spaced
    y = np.linspace(-3, 3, d)
    X, Y = np.meshgrid(x, y) #create the grid for the plot
    l = []
    for i in range(0,d):
        for k in range(0,d):
            l.append([X[i][k], Y[i][k]])
    l = np.array(l)
    ones = np.ones(l.shape[0])
    data = np.c_[ones, l].T
    Z = funct(data, W, V, ) #evaluate the function (note that X,Y,Z are matrix)

    ax.plot_surface(X, Y, Z.reshape(d,d), rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()



############## DERIVATE ####################
def derivate_E_wrt_v(W, X, y, p, V, rho):
    t = np.dot(W, X) # 20*186
    temp = np.tanh(t)   #20*186
    b = np.dot(V, temp) # (1,186)
    y = y.reshape(-1, 1)
    B = b.T - y
    return (1/p  * np.dot(temp, B) ) + rho * V.T

def derivate_g_wrt_a(W, x, N):
    t = np.dot(W, x)
    temp = 1 - np.tanh(t)**2 # (20, 186)
    tan = np.repeat(temp, 3, axis=0)          
    x_repeated = np.tile(x, (N,1))
    return x_repeated * tan

def derivate_E_wrt_a(W, x, y, V, p, rho, N):
    t = np.dot(W, x) # 20*186
    temp = g(t)   #20*186
    b = np.dot(V, temp) # (1,186)
    y = y.reshape(-1, 1)
    B = b.T - y
    d_g_a = derivate_g_wrt_a(W, x, N)
    V_repeated = V.repeat(3).reshape(-1,1)    
    W_as_vector = W.reshape(N*3,1)            

    return  1/p * V_repeated * np.dot(d_g_a, B) + rho * W_as_vector

def gradient(omega, args):
    # unpacket
    row_w, col_w, row_v, col_v, x, rho, y, N, reg = args 
    W = omega[: row_w * col_w ].reshape(row_w, col_w)
    V = omega[row_w * col_w:].reshape(row_v, col_v)
    # initialization
    wx = np.dot(W, x) # (200, 186) W @ x
    A = g(wx) # (20, 186)
    B = np.dot(V, A).T # (186, 1)
    p = x.shape[1]
    W_column = W.flatten().reshape(-1,1) # W in column
    d_e_a = derivate_E_wrt_a(W, x, y, V, p, rho,N)
    d_e_v = derivate_E_wrt_v(W, x, y, p, V, rho) # (W, X, y, p, V, rho)
    gradient = np.squeeze(np.append(d_e_a, d_e_v))
    return gradient




####### MODIFIED SKLEARN TO DID THE K-FOLD CROSS VALIDATION ##########

from sklearn.base import *
from sklearn.ensemble import *
import sys


class ResampledEnsemble2(BaseEstimator):
    def __init__(self, N=100, m=None, n=None, rho=None, seed=1804475, W=None, V= None, start=-1, end=1, bool_reg=True):
        self._estimator_type = "Predictor"
        self.N = N
        self.m = m
        self.n = n
        self.rho = rho
        self.seed = seed
        self.W = W
        self.V = V
        self.loss = 0
        self.start = start
        self.end = end
        self.bool_reg = bool_reg
        #self.sigma = sigma

    def fit(self, X, y):
        W = input_hidden_weights(self.N, self.n, self.seed, self.start, self.end)
        V = hidden_output_weights(self.m, self.N, self.seed, self.start, self.end)
        X = X.T
        args = [W.shape[0], W.shape[1], V.shape[0],V.shape[1], X, self.rho, y , self.N, True ]
        omega = np.vstack([W.reshape(-1,1), V.reshape(-1,1)])
        args[-1] = self.bool_reg
        omega_opt = sc.optimize.minimize(loss, omega, args=args, method='BFGS', jac=gradient, tol=0.0001, options={"maxiter":3000})['x']
        self.W = omega_opt[: W.shape[0] * W.shape[1]].reshape(W.shape[0], W.shape[1])
        self.V = omega_opt[W.shape[0] * W.shape[1]: ].reshape(V.shape[0], V.shape[1])
        self.score(X, y)
        return self
        

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value      
        return self

    def predict(self, X):
        return feedforward(X, self.W, self.V )

    def score(self, X, y):
        global  best_loss, best_rho, best_N, losses_val, losses_train, avg_loss_batch
        val = False
        if X.shape[0] != 3: # validation
            X = X.T # di default sklearn è stupido
            val = True
            losses_train.append(np.mean(np.array(avg_loss_batch)))
            avg_loss_batch = []
        else:
            pred_cap = self.predict(X)
            self.loss = np.sum((pred_cap.reshape(len(y),) - y)**2) / (2*len(y))
            avg_loss_batch.append(self.loss)
        pred_cap = self.predict(X)
        ris = np.sum((pred_cap.reshape(len(y),)- y)**2) / (2*len(y))
        if val:
            losses_val.append(ris)
            if best_loss > ris:
                best_loss = ris  
                best_rho = self.rho
                best_N = self.N
        return ris


def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))


def transform(x):
    ones_ = np.ones(x.shape[0], dtype=float)
    x = x.T
    x = np.insert(x, 0, ones_, axis=0)
    return x
