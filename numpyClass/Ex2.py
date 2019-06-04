import numpy as np
import matplotlib.pyplot as plt

N = 1000
M = 50000
Y = []

for i in range(M):
    X = np.random.random(N)
    Y.append(X.sum())

plt.hist(Y, bins = 1000)
plt.show()
