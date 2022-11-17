from function_31_The_Halves import * 

# set best parameter
filename = r"Dataset.csv"
N = 50
n = 2
m = 1
rho = 1e-05
seed = 1804475  ## seed and range from best of 2.1 
position_label = 2 # position of the label in the csv
size_train = 186

df = np.genfromtxt(filename, delimiter=",", usemask=False)

x_train, y_train, x_test, y_test = shuffle_N_split(df, position_label, size_train)
x_train_trans = transform(x_train) # added vector ones
x_test_trans  = transform(x_test)  # added vector ones


outer_iterations, y_train_pred, end, W_new, V_new, loss_train = train(x_train_trans, y_train, N, n, m, rho, seed)

y_test_pred = feedforward(x_test_trans, W_new, V_new) # predict


print("Number of Neurons chosen N=", N)
print("Sigma=", 1)
print("rho=", rho)
print("Optimization with quadratic function for V and W")
print("Number of outer iterations:", outer_iterations)


print("Time needed for the entire 2-block decomposition", end)
print("Train Error:", score(y_train_pred, y_train))
# Final Test Error
print("Error test:", score(y_test_pred, y_test))
print("Loss train", loss_train)

a = input("Clicca invio per visualizzare il plot")
plotting(feedforward,"",  W_new, V_new, 100)