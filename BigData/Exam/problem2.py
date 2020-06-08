import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# Data import
path = os.getcwd()
data = np.load(path + "/Data/clustering.npz")
X = data['X']

# Data processing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(np.shape(X_scaled))

# PCA, scale down dimensions to 50
pca = PCA(n_components=50)
pcaData = pca.fit_transform(X_scaled)
pcaVis = PCA(n_components=2)
pcaVis = pcaVis.fit_transform(X_scaled)

# TSNE
t_sne = TSNE(n_components=2)
X_embedded = t_sne.fit_transform(X_scaled)

# Removing outliers using Spectral clustering
# specClust = SpectralClustering()
# res = specClust.fit_predict(X_scaled)
# outliers = np.where(res != 0)[0]

# print(np.where(res != 0)[0])

# newX = np.delete(X, outliers, axis=1)
# print(np.shape(newX))

# Removing outliers using Isolation Forest
rev = IsolationForest().fit_predict(X_scaled)
outliers = np.where(rev == -1)[0]
newX = np.delete(X, outliers, axis=0)
X_embedded_new = np.delete(X_embedded, outliers, axis=0)

print(np.sum(rev == -1))

# KMeans clustering on the data without outliers
kmeans3 = KMeans(n_clusters=3).fit_predict(newX)
kmeans4 = KMeans(n_clusters=4).fit_predict(newX)

# Feature importance based on KMeans clusters
forest = ExtraTreesClassifier(n_estimators=250).fit(newX, kmeans3)

importances3 = forest.feature_importances_
std3 = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices3 = np.argsort(importances3)[::-1]

forest = forest.fit(newX, kmeans4)
importances4 = forest.feature_importances_
std4 = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices4 = np.argsort(importances4)[::-1]


# Clustering with silhouette scores, Org. Data
range_n_clusters = range(2, 12)
sil_scores = []

for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X_scaled)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

sns.set()

cmap = plt.cm.tab10
colors = cmap.colors[0:4]

plt.figure()
plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1])
plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')

plt.figure()
plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1])
plt.scatter(x=X_embedded[outliers, 0], y=X_embedded[outliers, 1], c='r')
plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')

classes = []
plt.figure()
plt.scatter(x=X_embedded_new[:,0], y=X_embedded_new[:,1], c=kmeans3, cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(x=X_embedded[outliers, 0], y=X_embedded[outliers, 1], c='y')
plt.title("KMeans clustering with (n_cluster={}) visaulized with TSNE".format(3), weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
for c in range(0, 3):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
outlierC =  mpatches.Patch(color='y', label="Outliers")
classes.append(outlierC)
plt.legend(handles=classes)


classes = []
plt.figure()
plt.scatter(x=X_embedded_new[:,0], y=X_embedded_new[:,1], c=kmeans4, cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(x=X_embedded[outliers, 0], y=X_embedded[outliers, 1], c='y')
plt.title("KMeans clustering with (n_cluster={}) visaulized with TSNE".format(4), weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
for c in range(0, 4):
    patch = mpatches.Patch(color=cmap.colors[c], label="Class {}".format(c))
    classes.append(patch)
classes.append(outlierC)
plt.legend(handles=classes)

plt.figure()
plt.scatter(x=pcaVis[:,0], y=pcaVis[:,1])
plt.scatter(x=pcaVis[outliers, 0], y=pcaVis[outliers, 1], c='r')
plt.title("PCA Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Prin Comp 1", weight='bold').set_fontsize('10')
plt.ylabel("Prin Comp 2", weight='bold').set_fontsize('10')

plt.figure()
plt.bar(x=range_n_clusters, height=sil_scores, color='g', width=0.6)
plt.xlabel("Number of clusters")
plt.ylabel("Avg. Silhouette score")
plt.title("Silhouette score")


# Clustering with silhouette scores, PCA Data
sil_scores = []
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(pcaData)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(pcaData, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(pcaData, cluster_labels)

plt.figure()
plt.bar(x=range_n_clusters, height=sil_scores, color='g', width=0.6)
plt.xlabel("Number of clusters")
plt.ylabel("Avg. Silhouette score")
plt.title("Silhouette score on PCA components")

# Clustering with silhouette scores, Data with removed outliers
sil_scores = []
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(newX)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(newX, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(newX, cluster_labels)

plt.figure()
plt.bar(x=range_n_clusters, height=sil_scores, color='g', width=0.6)
plt.xlabel("Number of clusters")
plt.ylabel("Avg. Silhouette score")
plt.title("Silhouette score on new X")

# Hierarchical clustering on data with removed outliers
agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
agglo = agglo.fit(newX)

plt.figure()
plt.title('Hierarchical Clustering Dendrogram for data without outliers')
plot_dendrogram(agglo, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

# Plot the impurity-based feature importances of the forest
nFeat = 10
plt.figure()
plt.title("Feature importances (n_clusters=3)")
plt.bar(range(nFeat), importances3[indices3[:nFeat]],
        color="r")
plt.xticks(range(nFeat), indices3)
plt.xlim([-1, nFeat])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (n_clusters=3)")
plt.scatter(x=range(len(importances3)), y=importances3[indices3], c='r')
plt.xticks([], [])
plt.xlim([-1, len(importances3)])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (n_clusters=4)")
plt.bar(range(nFeat), importances4[indices4[:nFeat]],
        color="r")
plt.xticks(range(nFeat), indices4)
plt.xlim([-1, nFeat])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (n_clusters=4)")
plt.scatter(x=range(len(importances4)), y=importances4[indices4], c='r')
plt.xticks([], [])
plt.xlim([-1, len(importances4)])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
fIx = 220
plt.scatter(x=range(len(X_scaled[:, fIx])), y=X_scaled[:, fIx])
plt.xlabel("Data point index")
plt.ylabel("Feature value")
plt.title("Feature value {} for all data points (scaled)".format(fIx))

plt.figure()
fIx = np.where(X_scaled == np.amax(X_scaled))[1]
plt.scatter(x=range(len(X_scaled[:, fIx])), y=X_scaled[:, fIx])
plt.xlabel("Data point index")
plt.ylabel("Feature value")
plt.title("Feature value {} for all data points (scaled)".format(fIx))

fig1, sub = plt.subplots(1, 5)
fig1.subplots_adjust(wspace=0.6, hspace=0.6)
binwidth = 1

for c in range(0, 3):
    for ix, ax in enumerate(sub.flatten()):
        data = newX[np.where(kmeans3==c)[0], indices3[ix]]
        ax.hist(data, color=colors[c], bins=np.arange(min(data), max(data) + binwidth, binwidth), density=True)
        ax.set_title("Feature {}".format(indices3[ix]))
        ax.set_xlabel("Feature values")
        ax.set_ylabel("Density")

fig1.legend(handles=classes[:-2])
fig1.suptitle("Univariant Histogram of 5 most important features across data (n_cluster=3)")

fig2, sub = plt.subplots(1, 5)
fig2.subplots_adjust(wspace=0.6, hspace=0.6)
binwidth = 1

for c in range(0, 4):
    for ix, ax in enumerate(sub.flatten()):
        data = newX[np.where(kmeans4==c)[0], indices4[ix]]
        ax.hist(data, color=colors[c], bins=np.arange(min(data), max(data) + binwidth, binwidth), density=True)
        ax.set_title("Feature {}".format(indices4[ix]))
        ax.set_xlabel("Feature values")
        ax.set_ylabel("Density")

fig2.legend(handles=classes[:-1])
fig2.suptitle("Univariant Histogram of 5 most important features across data (n_cluster=4)")

plt.show()