# testing the seaborn library (matplotlib +)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # Pretty much a prettier variant of MatplotLib

df = pd.read_csv('http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv')

print(df.head())

gear_counts = df['# Gears'].value_counts()
# gear_counts.plot(kind='bar')
# plt.show()


# Same plot but with seaborn

sns.set()   # Overriding matplotlib

gear_counts.plot(kind='bar')
plt.show()

sns.distplot(df['CombMPG'])
plt.show()

df2 = df[['Cylinders', 'CityMPG', 'HwyMPG', 'CombMPG']]
print(df2.head())

# sns.pairplot(df2, hue='Cylinders')
# plt.show()

sns.scatterplot(x='Eng Displ', y='CombMPG', data=df)
plt.show()
