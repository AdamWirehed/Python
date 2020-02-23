import numpy as np
import scipy.stats as stats
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
from fractions import Fraction
import pandas as pd

filepath = "ASCII_Comma/Chapter_8/bodytemp.txt"
df = pd.read_csv(filepath)

# Stratifying the data into two new datasets. One for males and the other for females
dfMale = df[df["gender"] == 1]
dfFemale = df[df["gender"] == 2]

# task a, i
xyBar = dfMale["temperature"].mean() - dfFemale["temperature"].mean()

s2Male = dfMale["temperature"].var()
s2Female = dfFemale["temperature"].var()

SE_xy = m.sqrt(s2Male/len(dfMale) + s2Female/len(dfFemale))
I_normal = [xyBar - 1.96*SE_xy, xyBar + 1.96*SE_xy]

print("Confidence interval (95%, normal): {}\n".format(I_normal))

# task ii - parametric test (assuming normal dist.) = two sample t-test

totSize = len(dfMale) + len(dfFemale)
s2Pooled = (len(dfMale) - 1)/(totSize - 2)*s2Male + (len(dfFemale) - 1)/(totSize - 2)*s2Female
degF = totSize - 2
t = 1.984   # Two sided t-test (95% confidence interval)
I_ttest = [xyBar - t*m.sqrt(s2Pooled)*(totSize/(len(dfMale)*len(dfFemale))), xyBar + t*m.sqrt(s2Pooled)*(totSize/(len(dfMale)*len(dfFemale)))]

print("Confidence interval (95%, t-test): {}\n".format(I_ttest))

# task iii, Rank sum test
dfRanked = df
dfRanked = dfRanked.sort_values('temperature')

sumMale = 0
sumFemale = 0

for index, row in dfRanked.iterrows():
    if row["gender"] == 1:
        sumMale += (index + 1)
    elif row["gender"] == 2:
        sumFemale += (index + 1)
    else:
        print("Data at index {} is weird \n".format(index))

print(sumMale)
print(sumFemale)
print(sumMale + sumFemale)
print((len(dfMale) + len(dfFemale)) * (len(dfMale) + len(dfFemale) + 1)/2)

sns.set()

# measurements = dfMale["temperature"]
# stats.probplot(measurements, dist="norm", plot=pylab)
# pylab.show()
