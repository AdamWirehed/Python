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


## Depression dataset split
dfDep = pd.read_csv(pathData + "/carData/Ginzberg.csv")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
ix_train, ix_test = next(kf.split(dfDep), None)
dfTrain = dfDep.iloc[ix_train]
dfTest = dfDep.iloc[ix_test]
xTrain = dfTrain['simplicity'].values
xTrain = xTrain.reshape(len(dfTrain), 1)
yTrain = dfTrain['depression'].values
yTrain = yTrain.reshape(len(dfTrain), 1)
xTest = (dfTest['simplicity'].values).reshape(len(dfTest), 1)
yTest = (dfTest['depression'].values).reshape(len(dfTest), 1)

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
TE_Dep_lin = sum(yTest - lin_regDep.predict(xTest))
TE_Dep_kNN = sum(yTest - kNN_regDep.predict(xTest))

print("Test error linear: {}".format(TE_Dep_lin))
print("Test error kNN: {}".format(TE_Dep_kNN))

""" cv = mod.ShuffleSplit(n_splits=1, test_size=0.2)
lin_reg = mod.cross_validate(lin_test, xDep, yDep, cv=cv)
kNN_reg = mod.cross_validate(kNN_test, xDep, yDep, cv=cv)
print(lin_cval['test_score'])
print(kNN_cval['test_score']) """

sns.set()
plt.scatter(xTrain, yTrain, color='red', label='Data points (training)')
plt.plot(xTrain, lin_regDep.predict(xTrain), color='blue', label='Linear regression')
plt.plot(sorted(xTrain), kNN_regDep.predict(sorted(xTrain)), color='green', label='k-neighbors (k={})'.format(k))
plt.xlabel("Simplicity - Measures subject's need to see the world in black and white")
plt.ylabel("Depression (Beck)")
plt.title("Regression on simplicity and depression")
plt.legend()
plt.savefig("Figures/depression.png")
plt.show()


plt.scatter(xTest, yTest, color='red', label='Data points (test)')
plt.plot(xTest, lin_regDep.predict(xTest), color='blue', label='Linear regression')
plt.plot(sorted(xTest), kNN_regDep.predict(sorted(xTest)), color='green', label='k-neighbors (k={})'.format(k))
plt.xlabel("Simplicity - Measures subject's need to see the world in black and white")
plt.ylabel("Depression (Beck)")
plt.title("Regression on simplicity and depression")
plt.legend()
plt.savefig("Figures/depression.png")
plt.show()

