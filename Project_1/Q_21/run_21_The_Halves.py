from function_21_The_Halves import * 

# set best parameter
filename = r"Dataset.csv"

rho = 1e-05
N = 50
sigma = 1
start = -2
end = 2
m = 1
n = 2 
seed = 1804475 

position_label = 2 # position of the label in the csv
size_train = 186

df = np.genfromtxt(filename, delimiter=",", usemask=False)
x_train, y_train, x_test, y_test = shuffle_N_split(df, position_label, size_train)
x_train_trans = transform(x_train) # added vector ones
x_test_trans  = transform(x_test)  # added vector ones

y_train_pred, end, W, V_new, G_train = train(x_train_trans, y_train, N, n, m, seed, rho, start, end)
g_test = transform_G(W, x_test_trans)
y_test_pred = feedforward(g_test, V_new) # predict


print("Number of Neurons chosen N=", N)
print("Sigma=", 1)
print("rho=", rho)
print("Optimization with quadratic_convex")
print("Time to optimize the network", end, "s")
print("Train Error:", score(y_train_pred, y_train))
# Final Test Error
print("Error test:", score(y_test_pred, y_test))
args = [G_train, rho, y_train, N, True]
print("Loss train", loss(V_new.reshape(-1,1), args))
a = input("Clicca invio per visualizzare il plot")
plotting(feedforward,"",  W, V_new, 100)