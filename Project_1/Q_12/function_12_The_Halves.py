import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time
from sklearn.base import *
from sklearn.ensemble import *
import itertools

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

def prediction_func(x,C,V,N,sigma):
    p = x.shape[1]
    res_phi = np.apply_along_axis(phi, 0, x,C,sigma,N,2).reshape(N,p)
    return np.dot(V,res_phi)

def CrossValidation(k, params):
    x, N, m, n, rho, start, end, seed, y, sigma = params
    C = create_C(N,n,seed,start,end)
    V = create_V(m,N,seed,start,end)
    p = x.shape[1]
    index_range = np.arange(p)
    index_range = index_range.reshape(k,int(p/k))
    list_loss = []
    list_scoring_train = []
    list_scoring_val = []
    omega = np.vstack([C.reshape(-1,1), V.reshape(-1,1)])
    for i in range(k):
        x_val = x[:,index_range[i]]
        x_train = x[:, np.delete(index_range, i, axis=0).flatten()]
        y_train = y[np.delete(index_range, i, axis=0).flatten()]
        y_val = y[index_range[i]]
        args = [x_train, N, sigma, rho, y_train, C.shape[0], C.shape[1], V.shape[0], V.shape[1], True]

        omega_opt = sc.optimize.minimize(loss, omega, args=args, method='BFGS', jac=gradient, tol=0.0001, options={"maxiter":3000})['x']

        new_C = omega_opt[: C.shape[0] * C.shape[1]].reshape(C.shape[0], C.shape[1])
        new_V = omega_opt[C.shape[0] * C.shape[1]: ].reshape(V.shape[0], V.shape[1])

        y_pred = prediction_func(x_train, new_C, new_V, N, sigma)
        y_pred_val = prediction_func(x_val, new_C, new_V, N, sigma)

        list_scoring_train.append(score(y_pred, y_train))
        list_loss.append(loss(omega_opt, args))

        list_scoring_val.append(score(y_pred_val, y_val))
        args = [x_val, N, sigma, rho, y_val, C.shape[0], C.shape[1], V.shape[0], V.shape[1], True]

    scoring_val = np.array(list_scoring_val).mean()
    scoring_train = np.array(list_scoring_train).mean()
    loss_train = np.array(list_loss).mean()

    print("N:", N)
    print("Rho", rho)
    print("seed", seed)
    print("sigma", sigma)
    print("Error train:", scoring_train)
    print("Error val:", scoring_val)
    print("Loss train:", loss_train)
    print("")
    print("")
    print("")
    return loss_train, scoring_train, scoring_val

def grid_search(iterables, cv, input_params):
    best_params = []
    x_train, y_train, start, end = input_params 
    n = 2
    m = 1
    loss_train, scoring_train, scoring_val, best_scoring_val = 100000, 100000, 10000, 100000
    for t in itertools.product(*iterables):
        seed, N, rho, sigma = t
        params = [x_train, N, m, n, rho, start, end, seed, y_train]
        loss_train, scoring_train, scoring_val = CrossValidation(cv, params)
        if  scoring_val < best_scoring_val:
            best_scoring_val = scoring_val
            best_params = [N, seed, rho]
    return best_params

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


""""
#example in how we find our paramters

list_N = [20,40, 50, 60, 70]
list_sigma = [1]
list_seed = [1883300, 1884475]  
rho_init = np.random.uniform(10**-5,10**-3, 2)
rho_init = np.append(rho_init, [1e-05, 1e-03])
iterables = [list_seed, list_N, rho_init, list_sigma]
x_train_trans = trasform(x_train)
x_test_trans = trasform(x_test)
start = 0
end = 1
input_params = x_train_trans, y_train, start, end  
best_params = grid_search(iterables, 3, input_params)


TO GET SOME USEFUL VALUES
    gradient_v = gradient(omega, args)
    norm_gradient_initial = np.sqrt(np.sum(gradient_v)**2)
    print("norm gradient initial", norm_gradient_initial)
    initial_pred_train = feedforward(x_train, C, V,  N, sigma)
    initial_error_train = score(initial_pred_train, y_train)
    initial_pred_test = feedforward(x_test, C, V,  N, sigma)
    initial_error_test = score(initial_pred_test, y_test)
    print("initial error train", initial_error_train)
    print("initial error test", initial_error_test)
    print("loss initial guess", loss(omega, args))
    gradient_final = gradient(optimized['x'], args)
    norm_gradient_final = np.sqrt(np.sum(gradient_final)**2)
    print("norm gradient final", norm_gradient_final)
"""
