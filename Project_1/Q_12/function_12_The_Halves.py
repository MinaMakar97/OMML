import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as q
import sklearn
from sklearn.model_selection import GridSearchCV
import time
from sklearn.base import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
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

def train(x_train, y_train, N, n, m, rho, seed, sigma, bool_reg):
        
    # generate W and V
    C = create_C(N, n, seed, 0, 1)
    V = create_V(m, N, seed, 0, 1)

    # union in the same vector W and V
    omega = np.vstack([C.reshape(-1,1), V.reshape(-1,1)])
    
    args = [x_train, N, sigma, rho, y_train, C.shape[0], C.shape[1], V.shape[0], V.shape[1], bool_reg]
    
    # find best parameter
    start_time = time.time()
    optimized = sc.optimize.minimize(loss, omega, args=args, method="BFGS", jac=gradient, tol=0.0001, options={"maxiter":3000})
    end = time.time() - start_time 

    
    # recalculate C and V
    C_new = optimized['x'][: C.shape[0] * C.shape[1]].reshape(C.shape[0], C.shape[1])
    V_new = optimized['x'][C.shape[0] * C.shape[1]: ].reshape(V.shape[0], V.shape[1])

    # Final Train Error
    y_train_pred = feedforward(x_train, C_new, V_new, N, sigma)
    
    return optimized, y_train_pred, end, C_new, V_new, args


# N: number of neurons in the first hidden layer
# n: the dimension of input, in this project is 2
# m : the dimension of output
def create_V(m, N, seedd, start, end):
    np.random.seed(seedd)
    return np.random.uniform(start, end,size=(m, N))            
  
def create_C(N, n, seedd, start, end):
    np.random.seed(seedd)
    return np.random.uniform(start, end,size=(n*N, 1))

def phi(x, c, sigma, N, n): ##### phi for a single sample
  x = x.reshape(-1,1)
  radius = x-c
  radius_neurons = np.reshape(radius,(N,n)) # each row is the radius related to one neuron
  norm = np.sum(radius_neurons**2,axis=1)
  return np.exp(-(norm/sigma**2)).reshape(-1,1)   

def feedforward(x, C, V, N, sigma):
    p = x.shape[1]
    res_phi = np.apply_along_axis(phi, 0, x, C, sigma, N, 2).reshape(N,p)
    return np.dot(V,res_phi)

def g(t):
    return np.tanh(t)



def loss(omega,args):
  # initialization
  x, N, sigma, rho, y, row_c, col_c, row_v, col_v, reg = args
  #print(row_c, col_c, row_v, col_v, omega.shape)
  C = omega[: row_c * col_c ].reshape(row_c, col_c)
  V = omega[row_c * col_c: ].reshape(row_v, col_v)
  p = x.shape[1]
  # get prediction
  res_phi = np.apply_along_axis(phi, 0, x,C,sigma,N,2).reshape(N,p)
  prediction = np.dot(V,res_phi)
  # loss
  norma_reg = np.sum(omega ** 2)
  l = np.sum((prediction - y)**2) #((186, 1), (186, 1))
  regularization_term = (rho/2 * norma_reg)
  return (1 / (2 * p)) * l  + regularization_term * reg





def derivate_E_wrt_v(A, B, p, rho, V):
    return 1/p * np.dot(A, B.T) + rho * V.T

def derivate_E_wrt_c(B, p, rho, C, F):
    bf_t = np.dot(B, F.T)
    return 1/p * bf_t.T + rho * C


def gradient(omega,args):
    x, N, sigma, rho, y, row_c, col_c, row_v, col_v,reg = args
    C = omega[: row_c * col_c ].reshape(row_c, col_c)
    V = omega[row_c * col_c: ].reshape(row_v, col_v)
    p = x.shape[1]

    # derivate wrt V
    A = np.apply_along_axis(phi, 0, x,C,sigma,N,2).reshape(N,p)
    prediction = np.dot(V,A)
    B = prediction - y 
    der_E_v = derivate_E_wrt_v(A,B,p,rho,V)

    # derivate wrt C
    V_t = V.T
    V_rep = np.repeat(V_t, 2, axis=0)
    V_rep_sample = np.repeat(V_rep, p, axis=1)
    A_rep = np.repeat(A, 2, axis=0)
    C_sample = np.repeat(C, p, axis=1)
    radius_col = x - C_sample 
    constant = 2/sigma**2
    radius_col_con = radius_col * constant
    F = V_rep_sample * A_rep * radius_col_con
    der_E_c = derivate_E_wrt_c(B,p,rho,C,F)
    # gradient
    gradient = np.squeeze(np.append(der_E_c, der_E_v))
    return gradient


####### MODIFIED SKLEARN TO DID THE K-FOLD CROSS VALIDATION ##########




class ResampledEnsemble2(BaseEstimator):
    def __init__(self, N=100, m=None, n=None, rho=None, seed=1804475, C=None, V= None, start=-1, end=1, sigma=1, bool_reg=True):
        self._estimator_type = "Predictor"
        self.N = N
        self.m = m
        self.n = n
        self.rho = rho
        self.seed = seed
        self.C = C
        self.V = V
        self.loss = 0
        self.start = start
        self.end = end
        self.bool_reg = bool_reg
        self.sigma = sigma

    def fit(self, X, y):
        C = create_C(self.N, self.n, self.seed, self.start, self.end)
        V = create_V(self.m, self.N, self.seed, self.start, self.end)
        X = X.T
        X = np.tile(X, (self.N,1))
        args = [X, self.N, self.sigma, self.rho, y, C.shape[0], C.shape[1], V.shape[0], V.shape[1], True ]
        omega = np.vstack([C.reshape(-1,1), V.reshape(-1,1)])
        args[-1] = self.bool_reg
        omega_opt = sc.optimize.minimize(loss, omega, args=args, method='BFGS', jac=gradient, tol=0.0001, options={"maxiter":3000})['x']
        self.C = omega_opt[: C.shape[0] * C.shape[1]].reshape(C.shape[0], C.shape[1])
        self.V = omega_opt[C.shape[0] * C.shape[1]: ].reshape(V.shape[0], V.shape[1])
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
        return feedforward(X, self.C, self.V, self.N, self.sigma )

    def score(self, X, y):
        global  best_loss, best_rho, best_N, losses_val, losses_train, avg_loss_batch, best_sigma
        val = False
        #print(X.shape)
        if X.shape[0] != 2*self.N: # validation
            X = X.T # di default sklearn the given in input by sklearn is different from the required
            X = np.tile(X, (self.N,1))
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
                best_sigma = self.sigma
        return ris


def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))


def transform(x, N):
    x = x.T
    x = np.tile(x, (N,1))
    return x

def plotting(funct, title, C, V, d, N, sigma):
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
    data = l.T
    data = np.tile(data, (N,1))
    Z = funct(data, C, V, N, sigma) #evaluate the function (note that X,Y,Z are matrix)

    ax.plot_surface(X, Y, Z.reshape(d,d), rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()
