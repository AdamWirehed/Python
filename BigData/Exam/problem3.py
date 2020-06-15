import numpy as np
import math as m
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def listcommon(testlist, biglist):

    return list(set(testlist) & set(biglist)) 

n = 5000
p = 50
s = 0.1
gamma = 1

corrAdp = []
wrongAdp = []
corrLas = []
wrongLas = []

for ix in range(20):

    # Data generation
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p)/m.sqrt(p), size=n)

    beta = np.zeros(p)
    beta[:5] = np.random.normal(loc=0, scale=1, size=int(p*s))

    beta = beta.reshape(-1, 1)

    sigma_e = m.sqrt(np.linalg.norm(np.dot(X, beta))*np.linalg.norm(np.dot(X, beta))/(n - 1))/gamma
    e = np.random.normal(loc=0, scale=sigma_e, size=n).reshape(-1, 1)

    SNR = np.linalg.norm(np.dot(X, beta))/np.linalg.norm(e)
    print("SNR value: {}".format(SNR))

    y_true = np.dot(X, beta) + e
    y_true = y_true.reshape(-1, 1)

    A_true = np.where(beta != 0)[0]


    # Parameter estimation

    if (p >= n):
        # Estimating beta using ridge
        ridCV = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 100, 200, 1000]).fit(X, y_true)
        beta_est = ridCV.coef_
    else:
        # Estimating beta using Ordinary Least Squares
        linReg = LinearRegression().fit(X, y_true)
        beta_est = linReg.coef_[0]


    # Weights
    gam_w = 2
    w = 1/np.abs(beta_est)**gam_w
    X_prim = np.empty(shape=np.shape(X))

    for c in range(np.shape(X)[1]):
        X_prim[:, c] = X[:, c]/w[c]

    lasPCV = LassoCV(max_iter=10000).fit(X_prim, y_true.ravel())
    betaP_las = lasPCV.coef_
    beta_adp = betaP_las/w
    lasCV = LassoCV(max_iter=10000).fit(X, y_true.ravel())

    coefAdp = beta_adp
    coefLas = lasCV.coef_

    coefAdpIX = np.where(coefAdp != 0)[0]
    coefLasIX = np.where(coefLas != 0)[0]

    # print(lasPCV.alpha_)

    # print(A_true)
    # print(np.where(coefAdp != 0)[0])
    # print(np.where(coefLas != 0)[0])

    resAdp = len(listcommon(list(A_true), list(coefAdpIX)))
    resLas = len(listcommon(list(A_true), list(coefLasIX)))

    corrAdp.append(resAdp)
    corrLas.append(resLas)
    wrongAdp.append(len(coefAdpIX) - resAdp)
    wrongLas.append(len(coefLasIX) - resLas)

print(corrAdp)
print(wrongAdp)
print(corrLas)
print(wrongLas)

sns.set()

plt.figure()
plt.plot(np.array(corrAdp)/5, label="Correct est. non-zero coef. (Adaptive)")
plt.plot(np.array(wrongAdp)/45, label="Incorrect est. non-zero coef. (Adaptive).")
plt.plot(np.array(corrLas)/5, label="Correct est. non-zero coef. (Regular)")
plt.plot(np.array(wrongLas)/45, label="Incorrect est. non-zero coef. (Regular).")
plt.ylabel("Ratio of correct/incorrect estimated coef.")
plt.xlabel("Iteration Index")
plt.title("Comparasion of Adaptive and Regular Lasso parameter estimation (n={}, γ={})".format(n, gam_w))
plt.legend()


plt.show()