import numpy as np
import os
import random
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns

# Data import
path = os.getcwd()
data = np.load(path + "/Data/clustering.npz")
print(list(data.keys()))
X = data['X']


# Data processing
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(np.shape(X))


# PCA, scale down dimensions to 50
pca = PCA(n_components=50)
pcaData = pca.fit_transform(X)
pcaVis = PCA(n_components=2)
pcaVis = pcaVis.fit_transform(X)

# TSNE
t_sne = TSNE(n_components=2)
X_embedded = t_sne.fit_transform(X)

# Clustering
K = 3
kmeans_obj = KMeans(n_clusters=K).fit(X)
kmeans = kmeans_obj.predict(X)

dataC0 = X[np.where(kmeans == 0)[0], :]
dataC1 = X[np.where(kmeans == 1)[0], :]
dataC2 = X[np.where(kmeans == 2)[0], :]

print(np.shape(dataC0))
print(np.shape(dataC1))
print(np.shape(dataC2))

# Feature importance for each cluster


# Plotting

sns.set()

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

cmap = plt.cm.tab10
colors = cmap.colors[:K]
classes = []

for c in range(0, K):
    patch = mpatches.Patch(color=colors[c], label="Cluster {}".format(c))
    classes.append(patch)

plt.figure()
plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1], c=kmeans, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
plt.legend(handles=classes)


plt.show()