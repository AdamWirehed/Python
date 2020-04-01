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

## Datasets

# /datasets/iris.csv    Petal.Length, Petal.Width | Species
# /DAAG/toycars.csv     angle, distance           | car
# /vcd/Hitters.csv      Errors, Putouts           | Positions
# /vcd/Suicide.csv      age, method               | sex

## Variables
dataset = "/DAAG/toycars.csv"
pathData = "/Users/adamwirehed/Documents/GitHub/Rdatasets/csv"
df = pd.read_csv(pathData + dataset)
expl = ['angle', 'distance']
tar = ['car']

X = df[expl].values.reshape(len(df), len(expl))

if type(df[tar].values[0][0]) == np.int64:
    Y = np.ravel(df[tar].values)
else:
    Y = pd.factorize(np.ravel(df[tar].values))[0]


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
cv = mod.StratifiedKFold(n_splits=5, shuffle=True)
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
titles = ['Logistic', "kNN (k={})".format(k), 'LDA', 'QDA']
models = [log_reg, kNN_reg, lda_reg, qda_reg]

for model, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, model, xx, yy, cmap=cm.Set3, alpha=0.8)
    ax.scatter(X[:,0], X[:,1], c=Y, cmap=cm.Set3, s=20, edgecolors='k')
    ax.set_xlabel(expl[0])
    ax.set_ylabel(expl[1])
    ax.set_title(title)

filename = dataset.replace('.csv', '').split('/')[-1] + ".png"
plt.savefig("Figures/" + filename)
print(filename)
plt.show()
