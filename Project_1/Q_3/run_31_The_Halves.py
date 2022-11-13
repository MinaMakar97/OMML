from function_31_The_Halves import * 

# set best parameter
filename = r"Dataset.csv"
N = 50
n = 2
m = 1
rho = 3.3758055537366024e-05
seed = 1804475  ## seed and range from best of 2.1 
position_label = 2 # position of the label in the csv
size_train = 186

df = np.genfromtxt(filename, delimiter=",", usemask=False)

x_train, y_train, x_test, y_test = shuffle_N_split(df, position_label, size_train)
x_train_trans = transform(x_train) # added vector ones
x_test_trans  = transform(x_test)  # added vector ones


optimizer_1, optimizer_2, y_train_pred, end, W_new, V_new, loss_train = train(x_train_trans, y_train, N, n, m, rho, seed)

y_test_pred = feedforward(x_test_trans, W_new, V_new) # predict


print("Number of Neurons chosen N=", N)
print("Sigma=", 1)
print("rho=", rho)
print("Optimization solver BFGS for V and W")
print("Number of function evaluations of V:", optimizer_1['nfev'])
print("Number of gradient evaluations of V:", optimizer_1['njev'])
print("Number of iterations of V:", optimizer_1['nit'])
print("Output message of V:", optimizer_1['message'])

print("Number of function evaluations of W:", optimizer_2['nfev'])
print("Number of gradient evaluations of W:", optimizer_2['njev'])
print("Number of iterations of W:", optimizer_2['nit'])
print("Output message of W:", optimizer_2['message'])

print("Time to optimize the network for both", end)
print("Train Error:", score(y_train_pred, y_train))
# Final Test Error
print("Error test:", score(y_test_pred, y_test))
print("Loss train", loss_train)

a = input("Clicca invio per visualizzare il plot")
plotting(feedforward,"",  W_new, V_new, 100)