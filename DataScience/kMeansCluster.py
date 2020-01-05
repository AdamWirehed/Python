# Testing K-Means Cluster method

import random
import numpy as np


# Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0),
            np.random.normal(ageCentroid, 2.0)])

    X = np.array(X)
    return X


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

data = createClusteredData(100, 8)

model = KMeans(n_clusters=8)

# Scaling the data
model = model.fit(scale(data))

# We can look at the clusters each data point was assigned to
print(model.labels_)

# plot
sns.set()
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=model.labels_.astype(np.float))
plt.show()
