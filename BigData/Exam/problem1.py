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
from sklearn.metrics import confusion_matrix
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

    # print(np.shape(ridgeCV.coef_))
    # print(np.sum(np.sum(ridgeCV.coef_ == 0)))

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

print(logResOrg)
print("Logistic regression: {} (org. Data)".format(np.mean(logResOrg)))
print("Ridge regression: {} (org. Data)".format(np.mean(ridgeResOrg)))

print("Logistic regression: {} (PCA dim.)".format(np.mean(logResPCA)))
print("Ridge regression: {} (PCA dim.) \n".format(np.mean(ridgeResPCA)))

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

resultLas = permutation_importance(logCV, X_train, y_train, n_repeats=10)
ind_perLas = np.argsort(resultLas.importances_mean)[::-1]

resultRid = permutation_importance(ridgeCV, X_train, y_train, n_repeats=10)
ind_perRid = np.argsort(resultRid.importances_mean)[::-1]

# Print the feature ranking
print("Feature ranking Lasso:")

for f in range(10):
    print("%d. feature %d (%f +- %f)" % (f + 1, ind_perLas[f], resultLas.importances_mean[ind_perLas[f]], resultLas.importances_std[ind_perLas[f]]))

print("\nFeature ranking Ridge:")

for f in range(10):
    print("%d. feature %d (%f +- %f)" % (f + 1, ind_perRid[f], resultRid.importances_mean[ind_perRid[f]], resultRid.importances_std[ind_perRid[f]]))


# Special test with only the 2/50 best features

nF = 50

lasRes50 = []
ridgeRes50 = []
lasParam50 = []
ridgeParam50 = []

for ix in range(10):
    logCV.fit(X_train[:, ind_perLas[2:nF]], y_train)
    ridgeCV.fit(X_train[:, ind_perLas[2:nF]], y_train)

    res2Las = logCV.predict(X_valid[:, ind_perLas[2:nF]])
    res2Rid = ridgeCV.predict(X_valid[:, ind_perLas[2:nF]])

    lasRes50.append(logCV.score(X_valid[:, ind_perLas[2:nF]], y_valid))
    ridgeRes50.append(ridgeCV.score(X_valid[:, ind_perLas[2:nF]], y_valid))

    lasParam50.append(logCV.C_[0])
    ridgeParam50.append(ridgeCV.alpha_)

print("Lasso 48: {}".format(np.mean(lasRes50)))
print("Ridge 48: {}".format(np.mean(ridgeRes50)))

print(np.mean(logCV.C_[0]))
print(np.mean(ridgeCV.alpha_))

# plt.figure()
# plt.title("Feature importances (forest)")
# plt.bar(range(10), importances[indices[:10]],
#         color="r", yerr=std[indices][:10], align="center")
# plt.xticks(range(10), indices)
# plt.xlabel("Importance metric")
# plt.ylabel("Feature index")
# plt.xlim([-1, 10])

nFeatLas = 10
nFeatRid = 20

plt.figure()
plt.title("Feature importances (permutation) Lasso")
plt.bar(range(nFeatLas), resultLas.importances_mean[ind_perLas[:nFeatLas]],
        color="g", yerr=resultLas.importances_std[ind_perLas[:nFeatLas]], align="center")
plt.xticks(range(nFeatLas), ind_perLas)
plt.xlim([-1, nFeatLas])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (permutation) Lasso")
plt.scatter(x=range(len(resultLas.importances_mean[ind_perLas])), y =resultLas.importances_mean[ind_perLas], c='g')
plt.xticks([], [])
plt.xlim([-1, nFeat])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (permutation) Ridge")
plt.bar(range(nFeatRid), resultRid.importances_mean[ind_perRid[:nFeatRid]],
        color="r", yerr=resultRid.importances_std[ind_perRid[:nFeatRid]], align="center")
plt.xticks(range(nFeatRid), ind_perRid)
plt.xlim([-1, nFeatRid])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.title("Feature importances (permutation) Ridge")
plt.scatter(x=range(len(resultRid.importances_mean[ind_perRid])), y =resultRid.importances_mean[ind_perRid], c='r')
plt.xticks([], [])
plt.xlim([-1, nFeat])
plt.ylabel("Importance metric")
plt.xlabel("Feature index")

plt.figure()
plt.scatter(X_valid[:, 232], X_valid[:, 323], c=y_valid, cmap=matplotlib.colors.ListedColormap(colors))
plt.title("Validation data plotted in the two most important features")
plt.xlabel("Most important feature")
plt.ylabel("2nd most important feature")
plt.legend(handles=classes)

logCV.fit(X_train, y_train)
ridgeCV.fit(X_train, y_train)

resLas = logCV.predict(X_valid)
resRid = ridgeCV.predict(X_valid)

confLas = confusion_matrix(y_valid, resLas)
confRid = confusion_matrix(y_valid, resRid)
conf2Las = confusion_matrix(y_valid, res2Las)
conf2Rid = confusion_matrix(y_valid, res2Rid)

plt.figure()
sns.heatmap(confLas, annot=True, fmt='d', yticklabels=["True Class 0", "True Class 1"], xticklabels=["Classified 0", "Classified 1"])
plt.title("Confusion Matrix - Lasso")

plt.figure()
sns.heatmap(confRid, annot=True, fmt='d', yticklabels=["True Class 0", "True Class 1"], xticklabels=["Classified 0", "Classified 1"])
plt.title("Confusion Matrix - Ridge")

plt.figure()
sns.heatmap(conf2Las, annot=True, fmt='d', yticklabels=["True Class 0", "True Class 1"], xticklabels=["Classified 0", "Classified 1"])
plt.title("Confusion Matrix - Lasso (2 feat.)")

plt.figure()
sns.heatmap(conf2Rid, annot=True, fmt='d', yticklabels=["True Class 0", "True Class 1"], xticklabels=["Classified 0", "Classified 1"])
plt.title("Confusion Matrix - Ridge (2 feat.)")

print(sum(y_valid == 0)/nVal)

plt.show()