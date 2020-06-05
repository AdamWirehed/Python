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
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
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

# Hierarchical clustering
agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
agglo = agglo.fit(X_scaled)

# Removing outliers using Spectral clustering
specClust = SpectralClustering()
res = specClust.fit_predict(X_scaled)
outliers = np.where(res != 0)[0]

print(np.where(res != 0)[0])

newX = np.delete(X, outliers, axis=1)
print(np.shape(newX))

# Clustering with silhouette scores
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


sns.set()

plt.figure()
plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1])
plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')

plt.figure()
plt.bar(x=range_n_clusters, height=sil_scores, color='g', width=0.6)
plt.xlabel("Number of clusters")
plt.ylabel("Avg. Silhouette score")
plt.title("Silhouette score")

plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(agglo, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.figure()
fIx = 220
plt.scatter(x=range(len(X_scaled[:, fIx])), y=X_scaled[:, fIx])
plt.title("Feature value {} for all data points (scaled)".format(fIx))

plt.show()