import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Based on example given in https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")

neigh = KNeighborsClassifier()
neigh.fit(train_in, train_out)

KNN_train_classification = []
for number in train_in:
    KNN_train_classification.append(neigh.predict([number])[0])

train_correct = np.where(KNN_train_classification == train_out)[0]
train_percentage = 100 * len(train_correct) / len(train_in)
print(f"{train_percentage:.2f}% from the train set is correctly classified using the K-Nearest-Neighbor method.")

KNN_test_classification = []
for number in test_in:
    KNN_test_classification.append(neigh.predict([number])[0])

test_correct = np.where(KNN_test_classification == test_out)[0]
test_percentage = 100 * len(test_correct) / len(test_in)
print(f"{test_percentage:.2f}% from the test set is correctly classified using the K-Nearest-Neighbor method.")
