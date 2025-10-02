import numpy as np
import math

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")

bias = np.ones((len(train_in), 1))
T = np.append(train_in, bias, axis=1)

# Construct onehot matrix with a 1 in each row corresponding to the correct output of the training data.
onehots = np.zeros((len(train_in), 10))

for i in range(len(train_in)):
    j = int(train_out[i])
    onehots[i][j] = 1

def compute_softmaxes(W):
    softmaxes = np.zeros((len(train_in), 10))

    for i in range(len(train_in)):
        for k in range(10):
            softmaxes[i][k] = math.e ** np.dot(W[k], np.transpose(T[i]))

        softmaxes[i] /= sum(softmaxes[i])
    
    return softmaxes

def compute_gradients(i, softmaxes):
    gradients = np.zeros((10, 257))
    for k in range(10):
        gradients[k] = 2 * T[i] * (softmaxes[i][k] - onehots[i][k])

    return gradients

step_size = 1

# Initialise the weights matrix with random values between -0.5 and 0.5
W = np.random.rand(10, 257)
W = np.subtract(W, np.ones((10, 257)) / 2)


# Training loop with epochs
for epoch in range(300):
    softmaxes = compute_softmaxes(W)
    total_gradients = np.zeros((10, 257))
    for i in range(len(train_in)):
        total_gradients += compute_gradients(i, softmaxes)

    total_gradients = total_gradients / len(train_in)

    W -= total_gradients * step_size
    print("Loss: " + str(-sum(sum(onehots * np.log(softmaxes))) / len(train_in)))

results = np.zeros(len(train_in))

# Validate the results for the training data

for i in range(len(train_in)):
    output = np.argmax(np.dot(W, np.transpose(T[i])))
    if output == train_out[i]:
        results[i] = 1
    else:
        results[i] = 0

print("Correct results for the training set: ")
print(float(sum(results)) / len(train_in))


# Validate the results for the test data
results = np.zeros(len(test_in))

bias = np.ones((len(test_in), 1))
T = np.append(test_in, bias, axis=1)

for i in range(len(test_in)):
    output = np.argmax(np.dot(W, np.transpose(T[i])))
    if output == test_out[i]:
        results[i] = 1
    else:
        results[i] = 0

print("Correct results for the test set: ")
print(float(sum(results)) / len(test_in))

