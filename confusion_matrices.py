# Based on example given in: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

import numpy as np
from sklearn.metrics import confusion_matrix
from NM_classifier import find_nearest_mean
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")
test_in = np.genfromtxt("test_in.csv", delimiter=",")
test_out = np.genfromtxt("test_out.csv", delimiter=",")
centers = np.genfromtxt("cloud_centers.csv", delimiter=",")

# We only fit the KNN classifier with the training data.
neigh = KNeighborsClassifier()
neigh.fit(train_in, train_out)

KNN_train_classification = []
for number in train_in:
    KNN_train_classification.append(neigh.predict([number])[0])

KNN_test_classification = []
for number in test_in:
    KNN_test_classification.append(neigh.predict([number])[0])

nearest_mean_train_classification = find_nearest_mean(train_in, centers)
nearest_mean_test_classification = find_nearest_mean(test_in, centers)

def plot_cf_matrix(ground_truth, prediction, title):
    cf_matrix = confusion_matrix(ground_truth, prediction)  # Ground truth values go first!
    cf_matrix = cf_matrix.astype(float)
    
    for i in range(cf_matrix.shape[0]):
        total = sum(cf_matrix[i, :])
        cf_matrix[i, :] = cf_matrix[i, :] / total
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='.1%', linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

plot_cf_matrix(train_out, nearest_mean_train_classification, "Confustion Matrix for NM Classification on Training Data")
plot_cf_matrix(train_out, KNN_train_classification, "Confustion Matrix for KNN Classification on Training Data")
plot_cf_matrix(test_out, nearest_mean_test_classification, "Confustion Matrix for NM Classification on Test Data")
plot_cf_matrix(test_out, KNN_test_classification, "Confustion Matrix for KNN Classification on Test Data")









plt.show()
