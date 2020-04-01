import numpy as np
import pandas as pd
import sklearn.linear_model as kit
import sklearn.neighbors as kNN
import sklearn.discriminant_analysis as disc
import sklearn.model_selection as mod
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# ----------------------------------------------------------------------

## Variables
pathData = "/Users/adamwirehed/Documents/GitHub/Rdatasets/csv"
df = pd.read_csv(pathData + "/DAAG/toycars.csv")
expl = ['angle', 'distance']
tar = ['car']
X = (df[expl].values).reshape(len(df), len(expl))
#Y = pd.factorize(df.Species)[0]
Y = np.ravel(df[tar].values)
print(Y)

# Logistic regression
log_reg = kit.LogisticRegression(random_state=0).fit(X, Y)
log_test = kit.LogisticRegression()

# k-neighbors
k = 8
kNN_reg = kNN.KNeighborsClassifier(n_neighbors=k)
kNN_reg.fit(X, Y)
kNN_test = kNN.KNeighborsClassifier(n_neighbors=k)

# Linear discirimant Analysis
lda_reg = disc.LinearDiscriminantAnalysis()
lda_reg.fit(X, Y)
lda_test = disc.LinearDiscriminantAnalysis()

# Qudratic discriminat analysis
qda_reg = disc.QuadraticDiscriminantAnalysis()
qda_reg.fit(X, Y)
qda_test = disc.QuadraticDiscriminantAnalysis()

# Cross-validation
cv = mod.KFold(n_splits=5, shuffle=True)
log_result = mod.cross_validate(log_test, X, Y, cv=cv)
kNN_result = mod.cross_validate(kNN_test, X, Y, cv=cv)
lda_result = mod.cross_validate(lda_test, X, Y, cv=cv)
qda_result = mod.cross_validate(qda_test, X, Y, cv=cv)

print(log_result['test_score'].mean())
print(kNN_result['test_score'].mean())
print(lda_result['test_score'].mean())
print(qda_result['test_score'].mean())


sns.set()
fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.6, hspace=0.6)
xx, yy = make_meshgrid(X[:,0], X[:,1])

plot_contours(sub[0, 0], log_reg, xx, yy, cmap=cm.Set3, alpha=0.8)
sub[0, 0].scatter(X[:,0], X[:,1], c=Y, cmap=cm.Set3, s=20, edgecolors='k')
sub[0, 0].set_xlabel(expl[0])
sub[0, 0].set_ylabel(expl[1])
sub[0, 0].set_title("Logistic")

plot_contours(sub[0, 1], kNN_reg, xx, yy, cmap=cm.Set3, alpha=0.8)
sub[0, 1].scatter(X[:,0], X[:,1], c=Y, cmap=cm.Set3, s=20, edgecolors='k')
sub[0, 1].set_xlabel(expl[0])
sub[0, 1].set_ylabel(expl[1])
sub[0, 1].set_title("kNN (k={})".format(k))


plot_contours(sub[1, 0], lda_reg, xx, yy, cmap=cm.Set3, alpha=0.8)
sub[1, 0].scatter(X[:,0], X[:,1], c=Y, cmap=cm.Set3, s=20, edgecolors='k')
sub[1, 0].set_xlabel(expl[0])
sub[1, 0].set_ylabel(expl[1])
sub[1, 0].set_title("LDA")

plot_contours(sub[1, 1], qda_reg, xx, yy, cmap=cm.Set3, alpha=0.8)
sub[1, 1].scatter(X[:,0], X[:,1], c=Y, cmap=cm.Set3, s=20, edgecolors='k')
sub[1, 1].set_xlabel(expl[0])
sub[1, 1].set_ylabel(expl[1])
sub[1, 1].set_title("QDA")

plt.savefig("Figures/Toycars.png")
plt.show()
