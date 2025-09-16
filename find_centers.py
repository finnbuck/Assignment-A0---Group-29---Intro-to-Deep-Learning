import numpy as np

def dist(x, y):
    z = x - y
    return np.linalg.norm(z)

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")

cloud = []

for i in range(0, 10):
    cloud.append(train_in[np.where(train_out == i)])


center = []
for j in range(0, 10):
    center.append(np.mean(cloud[j], axis=0))

np.savetxt("cloud_centers.csv", center, delimiter=",")

distance = np.zeros((10, 10))
for k in range(0, 10):
    for l in range(0, 10):
        distance[k, l] = dist(center[k], center[l])

print(distance)
