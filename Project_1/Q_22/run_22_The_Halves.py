from function_22_The_Halves import *

# set best parameter
filename = r"Dataset.csv"

rho = 1e-05
N = 60
sigma = 1
start = 0
end = 1
m = 1
n = 2
seed = 15615612

position_label = 2  # position of the label in the csv
size_train = 186

df = np.genfromtxt(filename, delimiter=",", usemask=False)
x_train, y_train, x_test, y_test = shuffle_N_split(
    df, position_label, size_train)


y_train_pred, V_new, args, execution_time, C = train(
    x_train, y_train, N, n, rho, seed, sigma)


y_test_pred = predict(x_test, C, sigma, N, n, V_new)


print("Number of Neurons chosen N=", N)
print("Sigma=", 1)
print("rho=", rho)
print("Optimization with Least-Squares solution")
print("Time to optimize the network", execution_time, "s")
print("Training Error:", score(y_train_pred, y_train))
# Final Test Error
print("Test Error:", score(y_test_pred, y_test))

print("Loss train", loss(V_new.reshape(-1, 1), args))
a = input("Click enter to visualize the plot")
plotting(prediction_func_plot, "",  C, V_new, 100, N, sigma)
