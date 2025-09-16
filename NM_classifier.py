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

train_nearest_mean = []
for i in range(len(train_in)):
    train_nearest_mean.append(dist_center(train_in[i]).item())

train_correct = np.where(train_nearest_mean == train_out)[0]
train_percentage = 100 * len(train_correct) / len(train_in)
print(f"{train_percentage:.2f}% from the train set is correctly classified using the nearest mean method.")

test_nearest_mean = []
for i in range(len(test_in)):
    test_nearest_mean.append(dist_center(test_in[i]).item())

test_correct = np.where(test_nearest_mean == test_out)[0]
test_percentage = 100 * len(test_correct) / len(test_in)
print(f"{test_percentage:.2f}% from the test set is correctly classified using the nearest mean method.")