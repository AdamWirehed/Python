import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import sklearn.datasets as data

N = 300
K = 2
F = 10
np.random.seed(0)

def linear_kernel(v):
    mat = np.matmul(v, v.T)
    return mat

def poly_kernel(v, scale, offset, m):
    mat = polynomial_kernel(v, gamma=scale, degree=m, coef0=offset)
    return mat

def RBF_kern(v, sigma=100):
    mat = rbf_kernel(X=v, gamma=1/(2*sigma**2))
    return mat

def laplacian_kern(v, alpha=1):
    mat = laplacian_kernel(v, gamma=alpha/F)
    return mat
    
def sigmoid_kern(v, alpha=1, coeff=1):
    mat = sigmoid_kernel(v, gamma=alpha/F, coef0=coeff)
    return mat


T = np.empty(shape=(N,K))

x = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
Z = np.zeros((N, K))
ixC = np.random.randint(low=0, high=K, size=(N,1))

for ix in range(0, N):
    Z[ix, ixC[ix]] = 1

n = np.zeros(shape=(K, 1))
#X = make_classification(n_samples=N, n_features=F, n_informative=F, n_redundant=0, n_repeated=0, n_classes=K, n_clusters_per_class=1, class_sep=3)
#X = data.make_moons(n_samples=N, noise=1/10)
X = data.make_circles(n_samples=N, noise=0.1, factor=0.1)
#X = data.make_checkerboard(shape=(2,2), n_clusters=1, noise=1/5)

X = X[0]

# X = np.zeros(shape=(N, K))
# half = int(N/2)
# X[:half, 0] = np.linspace(start=-10, stop=10, num=half)
# X[half:, 0] = np.linspace(start=-10, stop=10, num=half)
# X[:half, 1] = X[:half, 0]**3 + np.random.normal(loc=0, scale=10, size=(1, half))
# X[half:, 1] = X[half:, 0]**3 + np.random.normal(loc=0, scale=10, size=(1, half)) + 1000

print(X)

pca = PCA(n_components=2)
pComp = pca.fit_transform(X)

gramLin = linear_kernel(X)
deg = 10
gramPoly = poly_kernel(X, m=deg, scale=1, offset=1)
gramRBF = RBF_kern(X)
gramLap = laplacian_kern(X)
gramSig = sigmoid_kern(X)

gram = gramSig

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

classes = []
cmap = plt.cm.tab10
colors = cmap.colors[0:K]

sns.set()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='b')
plt.title("Generated data (n={})".format(N))
plt.ylabel("Feature 0")
plt.xlabel("Feature 1")


plt.figure()
plt.scatter(pComp[:, 0], pComp[:,1], c=minIx, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Clusterd centered data (n={}) with estimated cluster groups using Sigmoid".format(N, deg))
plt.ylabel("PCA 0")
plt.xlabel("PCA 1")
for c in range(0, K):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
plt.legend(handles=classes)
plt.legend(handles=classes)

plt.show()