import numpy as np

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")

bias = np.ones((1000, 1))
T = np.append(test_in, bias, axis=1)
