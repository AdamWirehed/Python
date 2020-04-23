import sklearn.datasets as dat
import sklearn.cluster as clust
import sklearn.mixture as mix
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


NFEAT = 8
NINFO = 5
NRED = 0
NREP = 0
NCLASS = 3
NCLUSTCLASS = 1

nUnrelated = NFEAT - NINFO - NRED - NREP

[X, Y] = dat.make_classification(n_samples=100, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=5, shuffle=False)

nUseful = NINFO + NRED + NREP
print("Useful features: first {}".format(NINFO))

sns.set()
binwidth = 1

fig1, sub = plt.subplots(2,int(NFEAT/2))
fig1.subplots_adjust(wspace=0.6, hspace=0.6)
colors = ['b', 'r', 'g', 'y', 'c']
classes = []

for c in range(0, NCLASS):
    patch = mpatches.Patch(color=colors[c], label="Class {}".format(c))
    classes.append(patch)
    for ix, ax in enumerate(sub.flatten()):
        data = X[Y==c,ix]
        ax.hist(data, color=colors[c], bins=np.arange(min(data), max(data) + binwidth, binwidth), density=True)
        ax.set_title("Feature {}".format(ix))
        ax.set_xlabel("Feature values")
        ax.set_ylabel("Density")

fig1.legend(handles=classes)
fig1.suptitle("Univariant Histogram of features across data")

fig2, sub = plt.subplots(2, 2)
fig2.subplots_adjust(wspace=0.6, hspace=0.6)
colors = ['b', 'r', 'g']
markers = ['o', '^', 's']
featurePairs = [[0, 1], [2, 3], [4, 5], [6, 7]]

for ix, ax in enumerate(sub.flatten()):
    fDim = featurePairs[ix]
    for c in range(0,3):
        dataX = X[Y==c, fDim[0]]
        dataY = X[Y==c, fDim[1]]
        ax.scatter(dataX, dataY, color=colors[c], marker=markers[c])
    
    ax.set_xlabel("Feature {}".format(fDim[0]))
    ax.set_ylabel("Feature {}".format(fDim[1]))

fig2.legend(handles=classes)
fig2.suptitle("2D Scatter Plot of data in different feature dimensions")
    

NFEAT = 1020
NINFO = 3
NRED = 0
NREP = 0
NCLASS = 3
NCLUSTCLASS = 1

uFeat = []
scoreKM = []
scoreHier = []
scoreGMM = []

USEDF = 4

[X, Y] = dat.make_classification(n_samples=1000, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=1, shuffle=False)

for ix in range(0, 50):
    kMeans = clust.KMeans(n_clusters=NCLASS*NCLUSTCLASS).fit_predict(X[:, 0:USEDF])
    Agglo = clust.AgglomerativeClustering(n_clusters=NCLASS*NCLUSTCLASS, affinity='euclidean').fit_predict(X[:, 0:USEDF])
    Gmm = mix.GaussianMixture(n_components=NCLASS).fit_predict(X[:, 0:USEDF])

    scoreKM.append(metrics.adjusted_rand_score(Y, kMeans))
    scoreHier.append(metrics.adjusted_rand_score(Y, Agglo))
    scoreGMM.append(metrics.adjusted_rand_score(Y, Gmm))
    uFeat.append(USEDF - NINFO - NRED - NREP)

    USEDF += 10

fig3 = plt.figure(3)
plt.plot(uFeat, scoreKM, label="K-Means")
plt.plot(uFeat, scoreHier, label="Agglomerative (hierarchical)")
plt.plot(uFeat, scoreGMM, label="GMM")
plt.title("Adjusted Rand Index for fixed {} informative features, {} classes".format(NINFO, NCLASS))
plt.ylabel("Adjusted Rand Score (0-1)")
plt.xlabel("# unrelated features")
plt.legend()
plt.show()

