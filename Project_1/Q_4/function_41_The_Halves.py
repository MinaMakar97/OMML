import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import torch
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



def feedforward(x, weights):
    output = x
    for i, w in enumerate(weights):
        output = torch.matmul(w, output) # W2(W1x)
        if i < len(weights) -1:
            output = g(output)
    return output

class Model_():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def params(self):
        for layer in self.layers:
            for param in layer.get_params():
                yield param

class Dense():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        torch.manual_seed(1795982)
        self.weights = torch.rand(out_dim, in_dim, dtype=torch.float32, requires_grad=True)
    
    def forward(self, x):
        first = torch.matmul(self.weights, x)
        return first 
    
    def get_params(self):
        return [self.weights]

class Tanh():
    def forward(self, x):
        return torch.tanh(x)

    def get_params(self):
        return []



def g(t):
    m = torch.nn.Tanh()
    return m(t)


def transform(x):
    ones_ = np.ones(x.shape[0], dtype=float)
    x = x.T
    x = np.insert(x, 0, ones_, axis=0)
    return torch.from_numpy(x.astype(np.float32))

def score(y_pred, y_true):
    return np.sum((y_pred.reshape(len(y_true),) - y_true)**2) / (2* len(y_true))


def plotting(funct, title, d ):
    #create the object empty figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #create the grid

    x = np.linspace(-2, 2, d) 
    y = np.linspace(-3, 3, d)
    X, Y = np.meshgrid(x, y) #create the grid for the plot
    l = []
    for i in range(0,d):
        for k in range(0,d):
            l.append([X[i][k], Y[i][k]])
    l = np.array(l)
    ones = np.ones(l.shape[0])
    data = np.c_[ones, l].T
    Z = funct(torch.from_numpy(data.astype(np.float32))) #evaluate the function (note that X,Y,Z are matrix)

    ax.plot_surface(X, Y, Z.detach().numpy().reshape(d,d), rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()


def train_N_predict(x_train, y_train, x_test, y_test, model, max_epochs, loss_function, learning_rate, patience):
        
    i = 0
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    i = 0
    pred_weight = np.array([])
    while i < max_epochs and patience > 0:
        y_pred = model.forward(x_train)
        loss = loss_function(y_pred.reshape(-1,1), torch.from_numpy(y_train.reshape(-1,1).astype(np.float32)))
        train_losses.append(loss.item())
        loss.backward()
        pred_weight = np.array([])
        after_weight = np.array([])
        with torch.no_grad():
            for weights in model.params():
                temp = weights.numpy().flatten()
                pred_weight = np.append(pred_weight, temp)
                weights -= learning_rate * weights.grad
                after_weight = np.append(after_weight, weights.numpy().flatten() )
            
            y_pred_test = model.forward(x_test)
            test_loss = loss_function(y_pred_test.reshape(-1,1), torch.from_numpy(y_test.reshape(-1,1).astype(np.float32)))
            test_losses.append(test_loss.item())

        diff_omega = np.linalg.norm(after_weight-pred_weight)
        tolerance = 10e-5
        if diff_omega < tolerance:
            patience -= 1
        else:
            patience = 10

        for weights in model.params():
            weights.grad.zero_()
        i += 1
       
    end = time.time() - start_time

    # Final Train Error
    y_train_pred = model.forward(x_train)
    loss_train = loss_function(y_train_pred.reshape(-1,1), torch.from_numpy(y_train.reshape(-1,1).astype(np.float32)))
    train_error = score(y_train_pred.detach().numpy(), y_train)

    # final Test Error
    y_test_pred = model.forward(x_test)
    loss_test = loss_function(y_test_pred.reshape(-1,1), torch.from_numpy(y_test.reshape(-1,1).astype(np.float32)))
    test_error = score(y_test_pred.detach().numpy(), y_test)

    return y_train_pred, y_test_pred, loss_train.item(), loss_test.item(), end, train_losses, test_losses, model, i, train_error, test_error