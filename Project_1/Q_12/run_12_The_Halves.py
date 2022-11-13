from function_12_The_Halves import * 

# set best parameter
filename = r"Dataset.csv"

rho = 1e-05
N = 50
sigma = 1
start = 0
end = 1
m = 1
n = 2 
seed = 1804475 
bool_reg = True

position_label = 2 # position of the label in the csv
size_train = 186

df = np.genfromtxt(filename, delimiter=",", usemask=False)

x_train, y_train, x_test, y_test = shuffle_N_split(df, position_label, size_train)
x_train_trans = transform(x_train, N) # added vector ones
x_test_trans  = transform(x_test, N)  # added vector ones

optimizer, y_train_pred, end, C_new, V_new, args = train(x_train_trans, y_train, N, n, m, rho, seed, sigma, bool_reg)

y_test_pred = feedforward(x_test_trans, C_new, V_new, N, sigma) # predict


print("Number of Neurons chosen N=", N)
print("Sigma=", 1)
print("rho=", rho)
print("Optimization solver BFGS")
print("Number of function evaluations: ", optimizer['nfev'])
print("Number of gradient evaluations: ", optimizer['njev'])
print("Number of iterations: ", optimizer['nit'])
print("Output message: ", optimizer['message'])
print("Time to optimize the network", end)
print("Train Error:", score(y_train_pred, y_train))
# Final Test Error
print("Error test:", score(y_test_pred, y_test))
print("Loss train", loss(optimizer['x'], args))

a = input("Clicca invio per visualizzare il plot")
plotting(feedforward,"",  C_new, V_new, 100, N, sigma)