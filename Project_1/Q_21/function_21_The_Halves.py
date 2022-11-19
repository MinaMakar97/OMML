import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as q
import time
from sklearn.base import *
from sklearn.ensemble import *

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

# N: number of neurons in the first hidden layer
# n: the dimension of input, in this project is 2
# m : the dimension of output
def create_V(m, N, seedd, start, end):
    np.random.seed(seedd)
    return np.random.uniform(start, end,size=(m, N))             
  
def create_W(N, n, seedd, start, end):
    np.random.seed(seedd)
    return np.random.uniform(start, end,size=(N, n + 1))

def train(x_train, y_train, N, n, m, seed, rho, start, end):
        
    # generate W and V
    W = create_W(N, n, seed, start, end)
    V = create_V(m, N, seed, 0, 1)
    G_train = transform_G(W, x_train)
    
    # find best parameter
    start_time = time.time()
    Q, C = quadratic_convex(G_train, V, y_train, len(y_train), rho)
    

    V_new = sc.linalg.lstsq(Q, C)[0]
    end = time.time() - start_time 
    V_optimized = V_new.T
    y_train_pred = feedforward(G_train, V_optimized)
    
    return y_train_pred, end, W, V_optimized, G_train


def transform_G(W, x):
    wx = np.dot(W, x)
    return g(wx)

def feedforward(G, V):
    output = np.dot(V,G)
    return output

def g(t):
    return np.tanh(t)



def loss(V, args):   
    if V.ndim == 1:
        V = V.reshape(len(V), 1)
    V = V.T
    G, rho, y, N, reg = args 
    norma = np.sum(V ** 2) 
    pred = np.dot(V, G).T # 186,1
    p = len(y) # 186
    regularization = (rho / 2) * norma 
    t1 = 1/(2*p)
    error = np.sum((pred.reshape(p, ) - y)**2)
    return t1 * error  + regularization * reg


def transform(x):
    ones_ = np.ones(x.shape[0], dtype=float)
    x = x.T
    x = np.insert(x, 0, ones_, axis=0)
    return x

#################### Derivate ######################

def derivate_E_wrt_v(G, y, p, V, rho):
    V = V.T
    b = np.dot(V, G) # (1,186)
    y = y.reshape(-1, 1)
    B = b.T - y
    return (1/p  * np.dot(G, B) ) + rho * V.T


def gradient(V, args):
    if V.ndim == 1:
        V = V.reshape(len(V), 1)
    # unpacket
    G, rho, y, N, reg = args  
    # initialization
    p = y.shape[0]
    d_e_v = derivate_E_wrt_v(G, y, p, V, rho) 
    return np.squeeze(d_e_v)


def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))


def quadratic_convex(G, V, y, p, rho):
    lambda_ = p * rho
    g_quad = np.dot(G, G.T)
    temp = lambda_ * np.eye(G.shape[0])
    Q = 1/len(y) * (g_quad + temp)
    C = (1/p) * np.dot(G, y.reshape(-1, 1))
    return Q , C

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

    G = transform_G(W, data)

    #Z = funct(data, W, V ) #evaluate the function (note that X,Y,Z are matrix)
    Z = funct(G, V)
    ax.plot_surface(X, Y, Z.reshape(d,d), rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()

'''
TO GET SOME USEFUL VALUES
    G_train = transform_G(W, x_train)
    G_test = transform_G(W, x_test)
    args = [G_train, rho, y_train, N, True]
    initial_pred_train = feedforward(G_train, V)
    initial_error_train = score(initial_pred_train, y_train)
    initial_pred_test = feedforward(G_test, V)
    initial_error_test = score(initial_pred_test, y_test)
    print("initial error train", initial_error_train)
    print("initial error test", initial_error_test)
    print("loss initial guess", loss(V.reshape(-1, 1), args))

'''
