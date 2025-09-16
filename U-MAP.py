import numpy as np
import scipy
import matplotlib.pyplot as plt
import umap

train_in = np.genfromtxt("train_in.csv", delimiter=",")
train_out = np.genfromtxt("train_out.csv", delimiter=",")

reducer = umap.UMAP()

embedding = reducer.fit_transform(train_in)

fig, ax = plt.subplots()
scatter = ax.scatter(embedding[:, 0],
                     embedding[:, 1],
                     c = train_out,
                     cmap = "Paired",
                     s = 5)
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()

