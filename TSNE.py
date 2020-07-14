import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg') # backend
import matplotlib.pyplot as plt


table = np.load('item_emb_1.npy')
# item_id = np.arange(1, table.shape[0]).tolist()

# item_emb = []
# for ids in range(1, table.shape[0]):
#     item_emb.append(table[ids, :])

item_id = [12888, 49583, 1, 4733, 5761, 10845, 11210, 26875, 37882, 39167, 39988, 43922,
45974, 47202, 47483, 47849, 48131, 48895, 52222, 34944, 50059, 4489, 5910, 11663, 12147,
12956, 32489, 37163, 44593, 48076, 54035, 3180, 4824, 14847, 17019, 31989, 33412, 40168,
45056, 48406, 54945, 47484]

item_emb = [table[ids, :] for ids in item_id]

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(item_emb)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))
for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(item_id[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()