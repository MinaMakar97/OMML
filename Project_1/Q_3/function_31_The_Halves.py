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

def train(x_train, y_train, N, n, m, rho, seed):
        
    # generate W and V
    W = input_hidden_weights(N, n, seed, -2, 2)
    V = hidden_output_weights(m, N, seed, 0, 1)

    # union in the same vector W and V
    omega = np.vstack([W.reshape(-1,1), V.reshape(-1,1)])

    start_time = time.time()
    ### repeated optimization with 2-block decomposition
    patience = 4 
    outer_iterations = 0
    total_nfev = 0
    total_njev = 0
    total_nit = 0
    last_message = ""
    tollerance = 0.01
    args = [W.shape[0], W.shape[1], V.shape[0], V.shape[1], x_train, rho, y_train.reshape(-1,1) , N, True]
    while patience > 0:
        prec_error = loss(omega, args)
        outer_iterations += 1
        wx = np.dot(W, x_train)
        G_train = g(wx)
        Q, C = quadratic_convex(G_train, V, y_train, len(y_train), rho)
        V_optimized = sc.linalg.lstsq(Q, C)[0]
        #### optimization wrt. W
        args_W = [W.shape[0], W.shape[1], V_optimized.reshape(1,-1), x_train, rho, y_train.reshape(-1,1) , N, True]   
        optimizer = sc.optimize.minimize(loss_W, W, args=args_W, method="BFGS", jac=gradient_W, tol=0.0001, options={"maxiter":3000})

        # used for the output
        total_nfev += optimizer['nfev']
        total_njev += optimizer['njev']
        total_nit += optimizer['nit']
        last_message = optimizer['message']
        
        W_optimized = optimizer['x']

        omega_new = np.vstack([W_optimized.reshape(-1,1), V_optimized.reshape(-1,1)]) ## (200,1)
        actual_error = loss(omega_new, args)
        
        W = W_optimized.reshape(W.shape[0], W.shape[1])
        V = V_optimized.reshape(1,-1)

        diff_omega = np.linalg.norm(omega_new-omega)
        tolerance = 10e-3
        if diff_omega < tolerance and actual_error - prec_error <= tollerance :
            patience -= 1
        else:
            patience = 4

        omega = omega_new
    end = time.time() - start_time
    # Final Train Error
    y_train_pred = feedforward(x_train, W, V)
    
    loss_train = loss(omega_new, args)

    return outer_iterations, y_train_pred, end, W, V, loss_train, total_nfev, total_njev, total_nit, last_message, optimizer, diff_omega


def hidden_output_weights(m, N, seedd, start, end):
    np.random.seed(seedd)
    return np.random.uniform(start, end,size=(m, N))             
  
def input_hidden_weights(N, n, seedd, start, end):
    np.random.seed(seedd)
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

def loss_V(V, args):   
    if V.ndim == 1:
        V = V.reshape(len(V), 1)
    V = V.T
    W, x, rho, y, N, reg = args 
    norma = np.sum(V ** 2) 
    #print(V.shape, G.shape)
    wx = np.dot(W, x) # (200, 186) W @ x         
    G = g(wx) 
    pred = np.dot(V, G).T # 186,1
    p = len(y) # 186
    #y = y.reshape(-1, 1) # 186, 1
    regularization = (rho / 2) * norma 
    t1 = 1/(2*p)
    error = np.sum((pred.reshape(p, ) - y)**2)
    #print(error)
    return t1 * error  + regularization * reg

def loss_W(W, args): 
    W_row, W_col, V, x, rho, y, N, reg = args 
    W = W.reshape(W_row,W_col)
    norma = np.sum(W ** 2)
    t = np.dot(W, x) # (200, 186) W @ x         
    temp = g(t) 
    pred = np.dot(V, temp).reshape(-1,1)      # V @ t 
    p = len(y)
    regularization = (rho / 2) * norma 
    t1 = 1/(2*p)
    error = np.sum((pred - y)**2)  #((186, 1), (186, 1))
    return t1 * error  + regularization * reg

def quadratic_convex(G, V, y, p, rho):
    lambda_ = p * rho
    g_quad = np.dot(G, G.T)
    temp = lambda_ * np.eye(G.shape[0])
    Q = 1/len(y) * (g_quad + temp)
    C = (1/p) * np.dot(G, y.reshape(-1, 1))
    return Q , C


############## DERIVATE ####################
def derivate_E_wrt_v(W, X, y, p, V, rho):
    V=V.reshape(-1,1).T
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

def gradient_V(V, args):
    # unpacket
    W, x, rho, y, N, reg = args 
    # initialization
    p = x.shape[1]
    #d_e_a = derivate_E_wrt_a(W, x, y, V, p, rho,N)
    d_e_v = derivate_E_wrt_v(W, x, y, p, V, rho) 
    gradient = np.squeeze(d_e_v)
    return gradient

def gradient_W(W, args):
    # unpacket
    W_row, W_col, V, x, rho, y, N, reg = args 
    # initialization
    W = W.reshape(W_row,W_col)
    p = x.shape[1]
    d_e_a = derivate_E_wrt_a(W, x, y, V, p, rho,N)
    gradient = np.squeeze(d_e_a)
    return gradient

def transform(x):
    ones_ = np.ones(x.shape[0], dtype=float)
    x = x.T
    x = np.insert(x, 0, ones_, axis=0)
    return x

def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))


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


'''
TO GET SOME USEFUL VALUES
    initial_pred_train = feedforward(x_train, W, V)
    initial_error_train = score(initial_pred_train, y_train)
    initial_pred_test = feedforward(x_test, W, V)
    initial_error_test = score(initial_pred_test, y_test)
    print("initial error train", initial_error_train)
    print("initial error test", initial_error_test)
    print("loss initial guess", loss(omega, args))
'''
