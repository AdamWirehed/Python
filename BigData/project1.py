import numpy as np
import pandas as pd
import sklearn.linear_model as kit
import sklearn.neighbors as kNN
from sklearn.model_selection import KFold
import sklearn.model_selection as mod
import matplotlib.pyplot as plt
import seaborn as sns

## Variables
pathData = "/Users/adamwirehed/Documents/GitHub/Rdatasets/csv"
df = pd.read_csv(pathData + "/carData/Ginzberg.csv")
expl = 'simplicity'
tar = 'depression'


## Depression dataset split
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
ix_train, ix_test = next(kf.split(df), None)
dfTrain = df.iloc[ix_train]
dfTest = df.iloc[ix_test]
xTrain = dfTrain[expl].values
xTrain = xTrain.reshape(len(dfTrain), 1)
yTrain = dfTrain[tar].values
yTrain = yTrain.reshape(len(dfTrain), 1)
xTest = (dfTest[expl].values).reshape(len(dfTest), 1)
yTest = (dfTest[tar].values).reshape(len(dfTest), 1)

# Linear regression
lin_regDep = kit.LinearRegression()
lin_regDep.fit(xTrain, yTrain)
lin_test = kit.LinearRegression()

# k-neighbors
k = 8
kNN_regDep = kNN.KNeighborsRegressor(n_neighbors=k)
kNN_regDep.fit(xTrain, yTrain)
kNN_test = kNN.KNeighborsRegressor(n_neighbors=k)

# Cross-validation
data = (df[expl].values).reshape(len(df), 1)
target = (df[tar].values).reshape(len(df), 1)
lin_reg = mod.cross_validate(lin_test, data, target, cv=4, return_train_score=True, return_estimator=True)
kNN_reg = mod.cross_validate(kNN_test, data, target, cv=4, return_train_score=True, return_estimator=True)

print(lin_reg['test_score'])
print(kNN_reg['test_score'])

sns.set()
plt.scatter(xTrain, yTrain, color='red', label='Data points (training)')
plt.plot(xTrain, lin_reg['estimator'][1].predict(xTrain), color='blue', label='Linear regression')
plt.plot(sorted(xTrain), kNN_regDep.predict(sorted(xTrain)), color='green', label='k-neighbors (k={})'.format(k))
plt.xlabel("Simplicity - Measures subject's need to see the world in black and white")
plt.ylabel("Depression (Beck)")
plt.title("Regression on simplicity and depression")
plt.legend()
plt.savefig("Figures/depression.png")
plt.show()


plt.scatter(xTest, yTest, color='red', label='Data points (test)')
plt.plot(xTest, lin_regDep.predict(xTest), color='blue', label='Linear regression')
plt.plot(sorted(xTest), kNN_reg['estimator'][1].predict(sorted(xTest)), color='green', label='k-neighbors (k={})'.format(k))
plt.xlabel("Simplicity - Measures subject's need to see the world in black and white")
plt.ylabel("Depression (Beck)")
plt.title("Regression on simplicity and depression")
plt.legend()
plt.savefig("Figures/depression.png")
plt.show()

