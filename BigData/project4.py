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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import sklearn.datasets as data

N = 300
K = 4
F = 2
np.random.seed(0)

def linear_kernel(v):
    mat = np.matmul(v, v.T)
    return mat

def poly_kernel(v, scale, offset, m):
    mat = polynomial_kernel(v, gamma=scale, degree=m, coef0=offset)
    return mat

def RBF_kern(v, gam):
    mat = rbf_kernel(X=v, gamma=gam)
    return mat

def laplacian_kern(v, alpha=1):
    mat = laplacian_kernel(v, gamma=alpha/F)
    return mat
    
def sigmoid_kern(v, alpha=1, coeff=1):
    mat = sigmoid_kernel(v, gamma=alpha/F, coef0=coeff)
    return mat


T = np.empty(shape=(N,K))

Z = np.zeros((N, K))
ixC = np.random.randint(low=0, high=K, size=(N,1))

for ix in range(0, N):
    Z[ix, ixC[ix]] = 1

n = np.zeros(shape=(K, 1))
#X = make_classification(n_samples=N, n_features=F, n_informative=F, n_redundant=0, n_repeated=0, n_classes=K, n_clusters_per_class=1, class_sep=1)
#X = data.make_moons(n_samples=N, noise=1/10)
X = data.make_circles(n_samples=int(N/2), noise=0.1, factor=0.1)
#X = data.make_checkerboard(shape=(2,2), n_clusters=1, noise=1/5)

X = X[0]

X2 = data.make_circles(n_samples=int(N/2), noise=0.1, factor=0.1)
X2 = X2[0]
X = np.vstack((X, X2))
X[int(N/2):, :] += 2

# X = np.zeros(shape=(N, K))
# half = int(N/2)
# X[:half, 0] = np.linspace(start=-10, stop=10, num=half)
# X[half:, 0] = np.linspace(start=-10, stop=10, num=half)
# X[:half, 1] = X[:half, 0]**3 + np.random.normal(loc=0, scale=10, size=(1, half))
# X[half:, 1] = X[half:, 0]**3 + np.random.normal(loc=0, scale=10, size=(1, half)) + 1000

# X[:, 0] = (X[:, 0] - np.mean(X[:, 0]))/np.std(X[:, 0])
# X[:, 1] = (X[:, 1] - np.mean(X[:, 1]))/np.std(X[:, 1])

#gramLin = linear_kernel(X)
deg = 3
gam = 2
#gramPoly = poly_kernel(X, m=deg, scale=1, offset=1)
gramRBF = RBF_kern(X, gam)
# gramLap = laplacian_kern(X)
# gramSig = sigmoid_kern(X)

gram = gramRBF

pca = KernelPCA(n_components=2, kernel='rbf', degree=deg, gamma=gam)
pComp = pca.fit_transform(X)

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
plt.title("Clusterd data (n={}) with estimated cluster groups using RBF kernel".format(N, deg))
plt.ylabel("PCA 0")
plt.xlabel("PCA 1")
for c in range(0, K):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
plt.legend(handles=classes)

plt.figure()
plt.scatter(X[:, 0], X[:,1], c=minIx, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Clusterd data (n={}) with estimated cluster groups using RBF kernel".format(N, deg))
plt.ylabel("Feature 0")
plt.xlabel("Feature 1")
plt.legend(handles=classes)

plt.show()