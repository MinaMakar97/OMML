import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.optimize as q
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

def compute_phi(x, c, sigma, N, n): ##### phi for a single sample
    x = x.reshape(-1,1)
    radius = x-c
    radius_neurons = np.reshape(radius,(N,n)) # each row is the radius related to one neuron
    norm = np.sum(radius_neurons**2,axis=1)
    return np.exp(-(norm/sigma**2)).reshape(-1,1)

# N: number of neurons in the first hidden layer
# m : the dimension of output
def create_C(N, n, x_train, seed):
    np.random.seed(seed)
    rang = np.arange(x_train.shape[0]) # list from 0 to 249
    np.random.shuffle(rang,) 
    point_selected = x_train[rang[0:N]]
    C = point_selected.reshape(N*n, 1)
    return C


# N: number of neurons in the first hidden layer
# n: the dimension of input, in this project is 2
def create_V(m, N, seed, start, end):
    np.random.seed(seed)
    return np.random.uniform(start, end,size=(m, N))    


def prediction_func(phi,V):
    return np.dot(V,phi)


def transform(x, N):
    x = x.T
    x = np.tile(x, (N,1))
    return x


def loss(V, args):
  # initialization
  V = V.T
  phi, rho, y, reg = args
  p = len(y)
  # get prediction
  prediction = np.dot(V,phi).T
  # loss
  norma_reg = np.sum(V ** 2)
  l = np.sum((prediction.reshape(p, ) - y)**2) #((186, 1), (186, 1))
  regularization_term = (rho/2 * norma_reg)
  return (1 / (2 * p)) * l  + regularization_term * reg



############## DERIVATE ####################
def derivate_E_wrt_v(A, B, p, rho, V):
  return 1/p * np.dot(A, B.T) + rho * V.T

def gradient(V, args): ####### is use if we apply the gradient instead of quadratic_convex
    # unpacket
    phi, rho, y, p = args  
    # derivate wrt V
    prediction = np.dot(V, phi)
    B = prediction - y 
    der_E_v = derivate_E_wrt_v(phi, B, p, rho, V)
    return np.squeeze(der_E_v)

###########################################

def quadratic_convex(phi, y, p, rho):
    lambda_ = p * rho
    g_quad = np.dot(phi, phi.T)
    temp = lambda_ * np.eye(phi.shape[0])
    Q = 1/len(y) * (g_quad + temp)
    C = (1/p) * np.dot(phi, y.reshape(-1, 1))
    return Q , C


def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))



########### plot and prediction part #############

def plotting(funct, title, C, V, d, N,sigma ):
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
    Z = prediction_func_plot(data, C, V, N, sigma) #evaluate the function (note that X,Y,Z are matrix)

    ax.plot_surface(X, Y, Z.reshape(d,d), rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()


def prediction_func_plot(x,C,V,N,sigma):
    p = x.shape[1]
    res_phi = np.apply_along_axis(compute_phi, 0, x,C,sigma,N,2).reshape(N,p)
    return np.dot(V,res_phi)


def predict(x_test, C, sigma, N, n, V):
    x_test_trans  = transform(x_test, N)  # added vector ones
    p_test = x_test.shape[0]
    phi_test = np.apply_along_axis(compute_phi, 0, x_test_trans, C, sigma, N, n).reshape(N, p_test)
    y_pred = prediction_func(phi_test, V) # prediction
    return y_pred


############### TRAINING PART #####################
def train(x_train, y_train, N, n, rho, seed, sigma):

    # generate C and V
    C = create_C(N, n, x_train, seed)
    x_train_trans = transform(x_train, N) # added vector ones

    p = x_train.shape[0]
    phi_train = np.apply_along_axis(compute_phi, 0, x_train_trans, C, sigma, N, n).reshape(N, p) # apply function(below) along axis zero for x_train
    args = [phi_train, rho, y_train, True]

    now = time.time()
    Q, C_quad = quadratic_convex(phi_train,  y_train, len(y_train), rho) # Calculate Q and C
    V_new = sc.linalg.lstsq(Q, C_quad)[0].T # find best V
    execution_time = time.time() - now

    # pred
    y_train_pred = prediction_func(phi_train, V_new)
    
    return y_train_pred, V_new, args, execution_time, C