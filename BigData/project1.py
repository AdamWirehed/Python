import numpy as np
import pandas as pd
import sklearn.linear_model as kit
import sklearn.neighbors as kNN
from sklearn.model_selection import train_test_split as tts
import sklearn.model_selection as mod
import matplotlib.pyplot as plt
import seaborn as sns

## Variables
pathData = "/Users/adamwirehed/Documents/GitHub/Rdatasets/csv"


## Depression dataset
dfDep = pd.read_csv(pathData + "/carData/Ginzberg.csv")
dfDepTrain, dfDepTest = tts(dfDep[['simplicity', 'depression']], test_size=0.2)
xDep = dfDepTrain['simplicity'].values
xDep = xDep.reshape(len(dfDepTrain), 1)
yDep = dfDepTrain['depression'].values
yDep = yDep.reshape(len(dfDepTrain), 1)

# Linear regression
lin_regDep = kit.LinearRegression()
lin_regDep.fit(xDep, yDep)
lin_test = kit.LinearRegression()

#k-neighbors
k = 8
kNN_regDep = kNN.KNeighborsRegressor(n_neighbors=k)
kNN_regDep.fit(xDep, yDep)
kNN_test = kNN.KNeighborsRegressor(n_neighbors=k)

xTest = (dfDepTest['simplicity'].values).reshape(len(dfDepTest), 1)
yTest = (dfDepTest['depression'].values).reshape(len(dfDepTest), 1)

cv = mod.ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
lin_cval = mod.cross_validate(lin_test, xDep, yDep, cv=cv)
kNN_cval = mod.cross_validate(kNN_test, xDep, yDep, cv=cv)
print(lin_cval['test_score'].mean())
print(kNN_cval['test_score'].mean())

sns.set()
plt.scatter(xDep, yDep, color='red', label='Data points')
plt.plot(xDep, lin_regDep.predict(xDep), color='blue', label='Linear regression')
plt.plot(sorted(xDep), kNN_regDep.predict(sorted(xDep)), color='green', label='k-neighbors (k={})'.format(k))
plt.xlabel("Simplicity - Measures subject's need to see the world in black and white")
plt.ylabel("Depression (Beck)")
plt.title("Regression on simplicity and depression")
plt.legend()
plt.savefig("Figures/depression.png")
plt.show()


