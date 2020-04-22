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

[X, Y] = dat.make_classification(n_samples=5000, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=5, shuffle=False)

nUseful = NINFO + NRED + NREP
print("Useful features: first {}".format(NINFO))

sns.set()
binwidth = 1

fig, sub = plt.subplots(2,int(NFEAT/2))
plt.subplots_adjust(wspace=0.6, hspace=0.6)
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

fig.legend(handles=classes)
fig.suptitle("Univariant Histogram of features across data")
plt.show()


NFEAT = 120
NINFO = 4
NRED = 0
NREP = 0
NCLASS = 5
NCLUSTCLASS = 1

uFeat = []
scoreKM = []
scoreHier = []
scoreGMM = []

USEDF = 8

[X, Y] = dat.make_classification(n_samples=5000, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=1.5, shuffle=False)

for ix in range(0, 20):
    kMeans = clust.KMeans(n_clusters=NCLASS*NCLUSTCLASS).fit_predict(X[:, 0:USEDF])
    Agglo = clust.AgglomerativeClustering(n_clusters=NCLASS*NCLUSTCLASS, affinity='euclidean').fit_predict(X[:, 0:USEDF])
    Gmm = mix.GaussianMixture(n_components=NCLASS).fit_predict(X[:, 0:USEDF])

    scoreKM.append(metrics.adjusted_rand_score(Y, kMeans))
    scoreHier.append(metrics.adjusted_rand_score(Y, Agglo))
    scoreGMM.append(metrics.adjusted_rand_score(Y, Gmm))
    uFeat.append(USEDF - NINFO - NRED - NREP)

    USEDF += 5

plt.plot(uFeat, scoreKM, label="K-Means")
plt.plot(uFeat, scoreHier, label="Agglomerative (hierarchical)")
plt.plot(uFeat, scoreGMM, label="GMM")
plt.title("Adjusted Rand index score for fixed 3 informative features, 5 classes")
plt.ylabel("Adjusted rand score (0-1)")
plt.xlabel("# unrelated features")
plt.legend()
plt.show()

