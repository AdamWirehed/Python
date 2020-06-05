import numpy as np
import os
import random
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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

def estimate_n_clusters(X):
   "Find the best number of clusters through maximization of the log-likelihood from EM."
   last_log_likelihood = None
   kf = KFold(n_splits=10, shuffle=True)
   components = range(50)[1:]
   for n_components in components:
       gm = GaussianMixture(n_components=n_components)

       log_likelihood_list = []
       for train, test in kf.split(X):
           gm.fit(X[train, :])
           if not gm.converged_:
               raise Warning("GM not converged")
           log_likelihood = -gm.score_samples(X[test, :])

           log_likelihood_list += log_likelihood.tolist()

       avg_log_likelihood = np.average(log_likelihood_list)
       print(avg_log_likelihood)

       if last_log_likelihood is None:
           last_log_likelihood = avg_log_likelihood
       elif avg_log_likelihood+10E-6 <= last_log_likelihood:
           return n_components-1
       last_log_likelihood = avg_log_likelihood

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

# Clustering
km_scores= []
km_silhouette = []
vmeasure_score =[]
db_score = []
for i in range(2, 12):
    km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
    preds = km.predict(X_scaled)
    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(X_scaled)))
    km_scores.append(-km.score(X_scaled))
    
    silhouette = silhouette_score(X_scaled,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(X_scaled,preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    print("-"*100)


specClust = SpectralClustering()
res = specClust.fit_predict(X_scaled)

agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
agglo = agglo.fit(X_scaled)

# nr_clust = estimate_n_clusters(X_scaled)
# print(nr_clust)

# Compute DBSCAN
db = DBSCAN(min_samples=10).fit(X_scaled)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# Feature importance for each cluster


# Plotting

sns.set()

range_n_clusters = range(2, 12)
sil_scores = []

for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_scaled)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)


plt.figure()
plt.title("The elbow method for determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=km_scores,s=150,edgecolor='k')
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("K-means score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)

plt.figure()
plt.scatter(x=range_n_clusters, y=sil_scores)
plt.xlabel("Number of clusters")
plt.ylabel("Avg. Silhouette score")
plt.title("Silhouette score")

plt.figure()
plt.scatter(x=pcaVis[:,0], y=pcaVis[:,1])
plt.title("PCA Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Prin Comp 1", weight='bold').set_fontsize('10')
plt.ylabel("Prin Comp 2", weight='bold').set_fontsize('10')

plt.figure()
plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1])
plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')

plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(agglo, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)

# cmap = plt.cm.tab10
# colors = cmap.colors[:K]
# classes = []

# for c in range(0, K):
#     patch = mpatches.Patch(color=colors[c], label="Cluster {}".format(c))
#     classes.append(patch)

# plt.figure()
# plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1], c=kmeans_org, cmap=matplotlib.colors.ListedColormap(colors))
# plt.title("TSNE Results: Soil | Clustering on Original data", weight='bold').set_fontsize('14')
# plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
# plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
# plt.legend(handles=classes)


plt.show()