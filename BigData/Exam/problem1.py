import numpy as np
import os
import random
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns

# Data import
path = os.getcwd()
data = np.load(path + "/Data/classification.npz")

# Preprocessing the data, we fit the data using the training data only
scaler = StandardScaler()
scaler.fit(data['X_train'])
X_train = scaler.transform(data['X_train'])
X_valid = scaler.transform(data['X_valid'])
y_train = data['y_train']
y_valid = data['y_valid']

nTrain = np.shape(X_train)[0]
nVal = np.shape(X_valid)[0]
nFeat = np.shape(X_train)[1]

# PCA analysis, analysis only done on traning data and transformed on both training and validation data
pca = PCA()
pca.fit(X_train)
pcaTrain = pca.transform(X_train)
pcaValid = pca.transform(X_valid)

print(np.shape(pcaTrain))

# Dataset is not balanced. More samples of class 0 than 1. But the representation in the validation set is similar
# to the representation in the training data
print(np.bincount(y_train))
print(np.bincount(y_valid))
print()


# Models, nr of folds = 5
logCV = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)  # Lasso-regualated Logistic regression
ridgeCV = RidgeClassifierCV(alphas=np.array([0.01, 0.1, 1, 100, 500, 1000, 5000, 10000]) ,cv=5)
forest = ExtraTreesClassifier(n_estimators=nFeat)

logResOrg = []
ridgeResOrg = []
logResPCA = []
ridgeResPCA = []

for ix in range(10):

    logCV.fit(X_train, y_train)
    ridgeCV.fit(X_train, y_train)

    # print(np.shape(logCV.coef_))
    # print(np.sum(np.sum(logCV.coef_ == 0)))

    print(np.shape(ridgeCV.coef_))
    print(np.sum(np.sum(ridgeCV.coef_ == 0)))

    # print("Log. Hyperparameter: λ = {:.4f}".format(logCV.C_[0]))
    # print("Ridge hyperparameter: α = {:.4f}".format(ridgeCV.alpha_))

    logResOrg.append(logCV.score(X_valid, y_valid))
    ridgeResOrg.append(ridgeCV.score(X_valid, y_valid))

    logCV.fit(pcaTrain, y_train)
    ridgeCV.fit(pcaTrain, y_train)

    # print("Log. Hyperparameter: λ = {:.4f}".format(logCV.C_[0]))
    # print("Ridge hyperparameter: α = {:.4f}".format(ridgeCV.alpha_))

    logResPCA.append(logCV.score(pcaValid, y_valid))
    ridgeResPCA.append(ridgeCV.score(pcaValid, y_valid))


print("Logistic regression: {} (org. Data)".format(np.mean(logResOrg)))
print("Ridge regression: {} (org. Data)".format(np.mean(ridgeResOrg)))

print("Logistic regression: {} (PCA dim.)".format(np.mean(logResPCA)))
print("Ridge regression: {} (PCA dim.) \n".format(np.mean(ridgeResPCA)))

# Special test with only the 2 best features

logCV.fit(X_train[:, [232, 323]], y_train)
ridgeCV.fit(X_train[:, [232, 323]], y_train)

print(logCV.score(X_valid[:, [232, 323]], y_valid))
print(ridgeCV.score(X_valid[:, [232, 323]], y_valid))

print(logCV.C_[0])
print(ridgeCV.alpha_)

# Plot and stuff
sns.set()
cmap = plt.cm.tab10
colors = [cmap.colors[0], cmap.colors[5]]
K = 2
classes = []

for c in range(0, K):
    patch = mpatches.Patch(color=colors[c], label="Class {}".format(c))
    classes.append(patch)

plt.figure()
plt.scatter(pcaValid[:, 0], pcaValid[:, 1], c=y_valid, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Data visalulized in two PCA-dimensions")
plt.xlabel("PCA 0")
plt.ylabel("PCA 1")
plt.legend(handles=classes)


# Feature importance
logCV.fit(X_train, y_train)
ridgeCV.fit(X_train, y_train)
forest.fit(X_train, y_train)

result = permutation_importance(logCV, X_train, y_train, n_repeats=10)
ind_per = np.argsort(result.importances_mean)[::-1]

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f +- %f)" % (f + 1, ind_per[f], result.importances_mean[ind_per[f]], result.importances_std[ind_per[f]]))

# plt.figure()
# plt.title("Feature importances (forest)")
# plt.bar(range(10), importances[indices[:10]],
#         color="r", yerr=std[indices][:10], align="center")
# plt.xticks(range(10), indices)
# plt.xlabel("Importance metric")
# plt.ylabel("Feature index")
# plt.xlim([-1, 10])

plt.figure()
plt.title("Feature importances (permutation)")
plt.bar(range(20), result.importances_mean[ind_per[:20]],
        color="g", yerr=result.importances_std[ind_per[:20]], align="center")
plt.xticks(range(20), ind_per)
plt.xlim([-1, 20])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.scatter(X_valid[:, 232], X_valid[:, 323], c=y_valid, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Validation data plotted in the two most important features")
plt.xlabel("Most important feature")
plt.ylabel("2nd most important feature")
plt.legend(handles=classes)

plt.show()