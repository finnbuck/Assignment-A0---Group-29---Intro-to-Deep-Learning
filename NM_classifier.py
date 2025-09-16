import numpy as np 

def dist(x, y):
    z = x - y
    return np.linalg.norm(z)
 
def dist_center(x):
    vector = np.tile(x, (10, 1))
    res = np.abs(vector - centers)
    abs = np.linalg.norm(res, axis=1)
    return np.where(abs == np.min(abs))[0][0]

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")
centers = np.genfromtxt("cloud_centers.csv", delimiter=",")

nearest_mean = []
for i in range(len(train_in)):
    nearest_mean.append(dist_center(train_in[i]).item())

correct = np.where(nearest_mean == train_out)[0]
percentage = 100 * len(correct) / len(train_in)
print(percentage)