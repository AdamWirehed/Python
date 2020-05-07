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
import matplotlib.patches as mpatches
import math as m

p, n, GAMMA = 80, 100, 11
nGroups = 5
nZeros = round(p/nGroups)*2
gSize = round(p/nGroups)

# Assigning beta parameters
beta = np.random.normal(loc=0, scale=1, size=p)

groups = np.ones(p)
F_groups = np.ones(p)
start = 0

for ix in range(0, nGroups):
    groups[start:start+gSize] = ix
    F_groups[start:start+gSize] = ix
    start += gSize

# Uncomment to give feature param. wrong groups
np.random.shuffle(F_groups)

print(groups)
print(F_groups)

# Generate data
X = np.ndarray((n,p), dtype=float)
cov_int = [0.1, 0.7, 0.2, 0.4, 0.3]     # Adds correlation between features within TRUE group

# Uncomment to have 0 correlation between features
#cov_int = np.zeros(p)

start = 0
for ig in range(0, nGroups):
    cov = (np.ones((gSize, gSize)) - np.eye(gSize))*cov_int[ig] + np.eye(gSize)
    X[:, start:start+gSize] = np.random.multivariate_normal(mean=np.zeros(gSize), cov=cov, size=n)
    start += gSize

sigma_e = m.sqrt(np.linalg.norm(np.dot(X, beta))*np.linalg.norm(np.dot(X, beta))/(n - 1))/GAMMA
e = np.random.normal(loc=0, scale=sigma_e, size=n)

SNR = np.linalg.norm(np.dot(X, beta))/np.linalg.norm(e)
print("SNR value: {}".format(SNR))

beta[0:nZeros] = 0   # First nZero features have no correlation to response
beta_bool = np.ndarray(shape=(p, 1), dtype=bool)
beta_bool[0:nZeros] = False
beta_bool[nZeros:] = True

Y = np.dot(X, beta) + e
Y = Y.reshape(-1, 1)

# l1_reg is the regularisation coeff. for coeffcient sparsiy pen.
# l2_reg is the regularisation coefficient(s) for the group sparsity penalty

gl = GroupLasso(
    groups=F_groups,
    l1_reg=0,
    group_reg=0.35,
    supress_warning=True
)

gl.fit(X, Y)
yhat = gl.predict(X)
beta_hat = gl.coef_

conf_m = confusion_matrix(beta_bool, gl.sparsity_mask_)

print("Number of variables: {}".format(p))
print("Number of zero coefficients: {}".format(nZeros))
print("Number of choosen variables: {}".format(gl.sparsity_mask_.sum()))
print(conf_m)

group_bool = np.ndarray(shape=(3, nGroups), dtype=int)
group_bool[0, :] = [0, 1, 2, 3, 4]
group_bool[1, :] = [nZeros/2, nZeros/2, 0, 0, 0]

Fgroup_bool = np.ndarray(shape=(3, nGroups), dtype=int)
Fgroup_bool[0, :] = [0, 1, 2, 3, 4]


# print(beta_hat[np.where(groups == 0)])
# print(beta_hat[np.where(groups == 1)])
# print(beta_hat[np.where(groups == 2)])
# print(beta_hat[np.where(groups == 3)])
# print(beta_hat[np.where(groups == 4)])

# print(beta_hat[np.where(F_groups == 0)])
# print(beta_hat[np.where(F_groups == 1)])
# print(beta_hat[np.where(F_groups == 2)])
# print(beta_hat[np.where(F_groups == 3)])
# print(beta_hat[np.where(F_groups == 4)])

for ig in range(0, nGroups):
    ix_g = np.where(groups == ig)
    ix_fg = np.where(F_groups == ig)
    sum_0_fg = sum(F_groups[:nZeros] == ig)   # In which false group are the true 0 at?
    nr_zeros = sum(gl.sparsity_mask_[ix_g] == False)
    nr_zeros_F = sum(gl.sparsity_mask_[ix_fg] == False)
    group_bool[2, ig] = nr_zeros
    Fgroup_bool[2, ig] = nr_zeros_F
    Fgroup_bool[1, ig] = sum_0_fg

print(gl.sparsity_mask_)

print()
print(group_bool)
print()
print(Fgroup_bool)

sns.set()

plt.figure()
plt.plot(beta, '.', label="True coefficients")
plt.plot(beta_hat, '.', label="Estimated coefficients")
plt.ylabel("Coeff. value")
plt.xlabel("Coeff. index")
plt.title("Group Lasso estimation for p={}, n={}, nZeros={}".format(p, n, nZeros))
plt.legend()

plt.figure()
plt.plot([beta.min(), beta.max()], [beta_hat.min(), beta_hat.max()], 'gray')
plt.scatter(beta, beta_hat, s=10)
plt.title("Group Lasso estimation for p={}, n={}, nZeros={}".format(p, n, nZeros))
plt.ylabel("Learned coefficients")
plt.xlabel("True coefficients")

plt.figure()
sns.heatmap(conf_m, annot=True, fmt='d', yticklabels=["True 0", "True non-0"], xticklabels=["Est. 0", "Est. non-0"])
plt.title("Confusion Matrix")

plt.figure()
sns.heatmap(group_bool[1:, :], annot=True, fmt='d', yticklabels=False)
plt.title("'Confusion' - (not really) matrix")
plt.xlabel("Group Index")
plt.ylabel("Est. # zeros          |         True # zeros")   

plt.figure()
sns.heatmap(Fgroup_bool[1:, :], annot=True, fmt='d', yticklabels=False)
plt.title("'Confusion' - (not really) matrix")
plt.xlabel("False Group Index")
plt.ylabel("Est. # zeros          |         True # zeros")   

plt.show()




