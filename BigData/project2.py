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

[X, Y] = dat.make_classification(n_samples=5000, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=10, shuffle=False)

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


NFEAT = 8
NINFO = 2
NRED = 0
NREP = 0
NCLASS = 4
NCLUSTCLASS = 1


[X, Y] = dat.make_classification(n_samples=5000, n_features=NFEAT, n_informative=NINFO, n_redundant=NRED, n_repeated=NREP, n_classes=NCLASS, n_clusters_per_class=NCLUSTCLASS, class_sep=10, shuffle=False)


kMeans = clust.KMeans(n_clusters=NCLASS*NCLUSTCLASS).fit_predict(X)
Agglo = clust.AgglomerativeClustering(n_clusters=NCLASS*NCLUSTCLASS, affinity='euclidean').fit_predict(X)
Gmm = mix.GaussianMixture(n_components=NCLASS).fit_predict(X)

print(metrics.fowlkes_mallows_score(Y, kMeans))
print(metrics.fowlkes_mallows_score(Y, Agglo))
print(metrics.fowlkes_mallows_score(Y, Gmm))
