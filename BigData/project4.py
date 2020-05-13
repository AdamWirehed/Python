import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.datasets import make_classification
import sklearn.datasets as data

N = 300
K = 5
F = 2
np.random.seed(0)

def linear_kernel(v):
    mat = np.matmul(v, v.T)
    return mat

def poly_kernel(v, scale=1, offset=1, m=2):
    mat = np.matmul((scale*v), v.T) + offset
    mat = mat**m
    return mat

def RBF_kern(v, sigma=100):
    mat = rbf_kernel(X=v, gamma=1/(2*sigma**2))
    return mat
    
def sigmoid_kern(v, alpha=1/F, coeff=1):
    mat = sigmoid_kernel(v, gamma=alpha, coef0=coeff)
    return mat


T = np.empty(shape=(N,K))

x = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
Z = np.zeros((N, K))
ixC = np.random.randint(low=0, high=K, size=(N,1))

for ix in range(0, N):
    Z[ix, ixC[ix]] = 1

n = np.zeros(shape=(K, 1))
#X = make_classification(n_samples=N, n_features=F, n_informative=2, n_redundant=0, n_repeated=0, n_classes=K, n_clusters_per_class=1, class_sep=1.5)
X = data.make_moons(n_samples=N, noise=1/10)
#X = data.make_circles(n_samples=N, noise=0.1, factor=0.1)



gramLin = linear_kernel(X[0])
gramPoly = poly_kernel(X[0], m=2)
gramRBF = RBF_kern(X[0])
gramSig = sigmoid_kern(X[0])

gram = gramPoly

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

# plt.figure()
# plt.scatter(X[0][:,0], X[0][:,1], c=X[1], cmap=matplotlib.colors.ListedColormap(colors))
# plt.title("Generated data with true cluster groups")
# plt.ylabel("Feature 0")
# plt.xlabel("Feature 1")

classes = []
cmap = plt.cm.tab10
colors = cmap.colors[0:K]
plt.figure()
plt.scatter(X[0][:,0], X[0][:,1], c=minIx, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Clusterd data with estimated cluster groups")
plt.ylabel("Feature 0")
plt.xlabel("Feature 1")
for c in range(0, K):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
plt.legend(handles=classes)
plt.legend(handles=classes)

plt.show()