import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import make_classification


def linear_kernal(v):
    mat = np.matmul(v, v.T)
    return mat

N = 150
K = 4
F = 2
np.random.seed(0)
T = np.empty(shape=(N,K))

x = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
Z = np.zeros((N, K))
ixC = np.random.randint(low=0, high=K, size=(N,1))
cmap = plt.cm.tab10
colors = cmap.colors[0:4]

for ix in range(0, N):
    Z[ix, ixC[ix]] = 1

n = np.zeros(shape=(K, 1))
X = make_classification(n_samples=N, n_features=F, n_informative=2, n_redundant=0, n_repeated=0, n_classes=K, n_clusters_per_class=1, class_sep=1.5)

gram = linear_kernal(X[0])
hell = True
it = 0

while hell:
    it += 1
    for i in range(0, K):
        n[i] = sum(Z[:, i])
        ones = np.ones((N, 1))
        Zcol = Z[:, i].reshape(-1, 1)
        ZcolT = Z[:, i].reshape(1, -1)

        # Split the operations into two rows for clarity AND A LOT OF RESHAPE...
        T[:, i] = (-(2 / n[i]) * np.sum(np.matmul(ones, ZcolT) * gram, axis=1)) 
        T[:, i] += np.squeeze((n[i] ** (-2) * np.sum(np.matmul(Zcol, ZcolT) * gram)) * ones)

    minIx = np.argmin(T, axis=1)
    prevZ = Z
    Z = np.zeros(shape=(N,K))

    for ix in range(0, N):
        Z[ix, minIx[ix]] = 1

    if np.all(Z == prevZ):
        hell = False

print("Number of iterations: {}".format(it))

sns.set()
plt.figure()
plt.scatter(X[0][:,0], X[0][:,1], c=X[1], cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Generated data with true cluster groups")
plt.ylabel("Feature 0")
plt.xlabel("Feature 1")

classes = []
for c in range(0, K):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
plt.legend(handles=classes)

plt.figure()
plt.scatter(X[0][:,0], X[0][:,1], c=minIx, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Clusterd data with estimated cluster groups")
plt.ylabel("Feature 0")
plt.xlabel("Feature 1")
plt.legend(handles=classes)

plt.show()