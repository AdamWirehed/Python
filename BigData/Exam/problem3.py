import numpy as np
import math as m

np.random.seed(0)

n = 55
p = 50
s = 0.1
gamma = 1

# Data generation
X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)

beta = np.zeros(p)
positions = np.random.choice(np.arange(p), int(p*s), replace=False)
beta[positions] = np.random.normal(loc=0, scale=1, size=int(p*s))

beta = beta.reshape(-1, 1)

sigma_e = m.sqrt(np.linalg.norm(np.dot(X, beta))*np.linalg.norm(np.dot(X, beta))/(n - 1))/gamma
e = np.random.normal(loc=0, scale=sigma_e, size=n)

SNR = np.linalg.norm(np.dot(X, beta))/np.linalg.norm(e)
print("SNR value: {}".format(SNR))

y_true = np.dot(X, beta) + e
y_true = y_true.reshape(-1, 1)

A_true = np.where(beta != 0)[0]
print(A_true)