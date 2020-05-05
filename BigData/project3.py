import numpy as np
import scipy as sp
import scipy.stats as stats
from group_lasso import GroupLasso
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math as m

p, n, GAMMA = 300, 1000, 11
nGroups = 10
nZeros = round(p/nGroups)
gSize = round(p/nGroups)
print(gSize)

beta = np.random.normal(loc=0, scale=1, size=p)
X = np.random.normal(loc=0, scale=1, size=n*p)
X = X.reshape((n, p))

sigma_e = m.sqrt(np.linalg.norm(np.dot(X, beta))*np.linalg.norm(np.dot(X, beta))/(n - 1))/GAMMA
e = np.random.normal(loc=0, scale=sigma_e, size=n)

SNR = np.linalg.norm(np.dot(X, beta))/np.linalg.norm(e)
print("SNR value: {}".format(SNR))

# Here we have the 1st 3rd of the data to be non-correlated
beta[0:nZeros] = 0   # First nZero features have no correlation to response
beta_bool = np.ndarray(shape=(p, 1), dtype=bool)
beta_bool[0:nZeros] = False
beta_bool[nZeros:] = True

Y = np.dot(X, beta) + e
Y = Y.reshape(-1, 1)

groups = np.ones(p)
groups[0:nZeros] = 0
start = nZeros

for ix in range(1, nGroups):
    groups[start:start+gSize] = ix
    start += gSize

# Uncomment to give feature param. wrong groups
groups = np.random.randint(0, nGroups, p)
print(groups)
print(beta_bool)

gl = GroupLasso(
    groups=groups,
    supress_warning=True
)

gl.fit(X, Y)
yhat = gl.predict(X)
beta_hat = gl.coef_

R2 = r2_score(Y, yhat)
conf_m = confusion_matrix(beta_bool, gl.sparsity_mask_)

print("Number of variables: {}".format(p))
print("Number of zero coefficients: {}".format(nZeros))
print("Number of choosen variables: {}".format(gl.sparsity_mask_.sum()))
print("R^2: {}".format(R2))
print(conf_m)

sns.set()
# df = pd.DataFrame(X)
# plt.figure()
# axes = pd.plotting.scatter_matrix(df)

plt.figure()
plt.plot(beta, '.', label="True coefficients")
plt.plot(beta_hat, '.', label="Estimated coefficients")
plt.legend()

plt.figure()
plt.plot([beta.min(), beta.max()], [beta_hat.min(), beta_hat.max()], 'gray')
plt.scatter(beta, beta_hat, s=10)
plt.ylabel("Learned coefficients")
plt.xlabel("True coefficients")


plt.show()






