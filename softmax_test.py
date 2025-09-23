import numpy as np
import math

train_in = np.genfromtxt("short_in.csv", delimiter=",")
train_out = np.genfromtxt("short_out.csv", delimiter=",")

bias = np.ones((len(train_in), 1))
T = np.append(train_in, bias, axis=1)

onehots = np.zeros((len(train_in), 10))

for i in range(len(train_in)):
    j = int(train_out[i])
    onehots[i][j] = 1

def z(k, i, W):
    z = 0
    for j in range(256):
        z += W[k][j] * T[i][j]
    return z

def compute_softmaxes(W):
    softmaxes = np.zeros((1707, 10))

    for i in range(len(train_in)):
        for k in range(10):
            softmaxes[i][k] = math.e ** z(k, i, W)

        softmaxes[i] /= sum(softmaxes[i])
    
    return softmaxes

def compute_gradients(i, W):
    gradients = np.zeros((10, 257))
    softmaxes = compute_softmaxes(W)
    for k in range(10):
        for j in range(257):
            gradients[k][j] = (softmaxes[i][k] - onehots[i][k]) * T[i][j]
    return gradients

step_size = 10
W = np.random.rand(10, 257)
W = np.subtract(W, np.ones((10, 257)) / 2)
print(W)

for epoch in range(100):
    total_gradients = np.zeros((10, 257))
    for i in range(len(train_in)):
        total_gradients += compute_gradients(i, W)
        # print("Alive at: " + str(i))
    total_gradients = total_gradients / len(train_in)

    W -= total_gradients * step_size
    print(sum(sum(W)))



