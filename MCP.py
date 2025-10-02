import numpy as np

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")

bias = np.ones((1707, 1))
y_bias = np.tile(train_out, (10, 1)).T
alpha = 1E-7
del_loss = np.zeros((1707, 257, 10))
T = np.append(bias, train_in, axis=1)
W = np.ones((257, 10))/100

for k in range(200):
    Z1 = np.dot(T, W)
    Z2 = Z1 - y_bias
    Z3 = 2 * Z2
    for i in range(len(train_out)):
        del_loss[i, 0, :] = Z3[i]
        del_loss[i, 1:, :] = np.dot(train_in[i].reshape(-1,1), np.expand_dims(Z3[i], axis=0))
    del_Loss = np.sum(del_loss, axis=0)
    W = W - alpha * del_Loss
    print(np.mean(del_Loss))
print(W)