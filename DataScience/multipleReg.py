import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

df1 = df[['Mileage', 'Price']]   # New seperate dataframe
bins = np.arange(0, 50000, 10000)
groups = df1.groupby(pd.cut(df1['Mileage'], bins)).mean()
# print(groups.head())

sns.set()

groups['Price'].plot.line()
plt.xticks(range(0, 4), bins)
plt.ylabel('Price')
plt.show()

scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(
                            X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

# scale.fit_transform transforms the values for Milage, Cylinder and Doors
# to values between -1 and 1 (normalized data)
# However there are values < -1 and > 1

print(X.head())

est = sm.OLS(y, X).fit()
print(est.summary())

print(y.groupby(df.Doors).mean())

# Mileage: 45000, Cyl: 8, Doors: 4, transformed based on previous data
statsCar = scale.transform([[45000, 8, 4]])
print(statsCar)
predicted = est.predict(statsCar[0])
print(predicted)
