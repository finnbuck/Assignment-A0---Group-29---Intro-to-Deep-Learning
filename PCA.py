import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler # Preprocessing recommended by: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-with-python/
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Based on example in: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py



train_in = pd.read_csv('train_in.csv')
train_out = pd.read_csv('train_out.csv')

# Preprocessing the scale
sc = StandardScaler()
train_in = sc.fit_transform(train_in)

X_reduced = PCA(n_components=2).fit_transform(train_in)

fig, ax = plt.subplots()

PCA_reduced = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=train_out.to_numpy(), alpha=0.6, label=train_out.to_numpy(), cmap="Paired")

legend = ax.legend(*PCA_reduced.legend_elements(num=10),
                    loc="upper right", title="Number")
ax.add_artist(legend)

plt.show()
