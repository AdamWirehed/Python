# Test-file for the "pandas" library

import matplotlib.pyplot as plt
import pandas as pd
import os

dirFile = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(dirFile + '/DataScience_CourseMaterial/PastHires.csv')
df.head()
# print(dataframe.head())
# print(dataframe.tail())
# print(dataframe['Employed?'])

# print(df.sort_values(['Years Experience']))
degree_count = df['Level of Education'].value_counts()
# print(degree_count)
# degree_count.plot(kind='bar')
# plt.show()


# Task from Frank
newDF = df[5:11][['Previous employers', 'Hired']]   # Nice trick
print(newDF)
preEmp_count = newDF['Previous employers'].value_counts()
print(preEmp_count)
preEmp_count.plot(kind='bar')
plt.show()