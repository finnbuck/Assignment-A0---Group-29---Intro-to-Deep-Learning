from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")

train_in_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=10).fit_transform(train_in)

fig, ax = plt.subplots()

PCA_reduced = ax.scatter(train_in_embedded[:,0], train_in_embedded[:,1], c=train_out, alpha=0.6, label=train_out, cmap="Paired")

legend = ax.legend(*PCA_reduced.legend_elements(num=10),
                    loc="upper right", title="Number")
ax.add_artist(legend)

plt.show()