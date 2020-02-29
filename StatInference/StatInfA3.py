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
t_obs = xyBar/m.sqrt(s2Pooled) * m.sqrt(totSize/(len(dfMale) * len(dfFemale)))

print("t-value: {}".format(t_obs))
print("Df: {}".format(degF))
print("p-value: {} = Not significant \n".format(stats.t.pdf(t_obs, degF)))


# task iii, Rank sum test
dfRanked = df
dfRanked = dfRanked.sort_values('temperature')

sumMale = 0
sumFemale = 0
rank = 0

for index, row in dfRanked.iterrows():
    if row["gender"] == 1:
        sumMale += (rank + 1)
    elif row["gender"] == 2:
        sumFemale += (rank + 1)
    else:
        print("Data at index {} is weird \n".format(index))
    rank += 1

print("Rank sum male: {}".format(sumMale))
print("Rank sum female: {}".format(sumFemale))
print("Fraction male sample vs female sample: {}".format(len(dfMale)/len(dfFemale)))
print("Normalized rank sum male: {}".format(sumMale * 1/(sumMale + sumFemale) * 20*(20+1)/2))
print("Normalized rank sum female: {}".format(sumFemale * 1/(sumMale + sumFemale) * 20*(20+1)/2))
print("Rank sum for males and females is not lower or higher than their bounds, Tl = 79, Tu = 131. Accept H0\n")

# print(sumMale)
# print(sumFemale)
# print(sumMale + sumFemale)
# print((len(dfMale) + len(dfFemale)) * (len(dfMale) + len(dfFemale) + 1)/2)


# task b, i

xyBarRate = dfMale["rate"].mean() - dfFemale["rate"].mean()

s2MaleRate = dfMale["rate"].var()
s2FemaleRate = dfFemale["rate"].var()

SE_xyRate = m.sqrt(s2MaleRate/len(dfMale) + s2FemaleRate/len(dfFemale))
I_normalRate = [xyBarRate - 1.96*SE_xyRate, xyBarRate + 1.96*SE_xyRate]

print("Confidence interval (95%, normal): {}\n".format(I_normalRate))


# b,ii - parametric test, two sample t-test

totSize = len(dfMale) + len(dfFemale)
s2PooledRate = (len(dfMale) - 1)/(totSize - 2)*s2MaleRate + (len(dfFemale) - 1)/(totSize - 2)*s2FemaleRate
degF = totSize - 2
t_obsRate = xyBarRate/m.sqrt(s2PooledRate) * m.sqrt(totSize/(len(dfMale) * len(dfFemale)))

print("t-value: {}".format(t_obs))
print("Df: {}".format(degF))
print("p-value: {} = Not significant \n".format(stats.t.pdf(t_obsRate, degF)))

dfRankedRate = df
dfRankedRate = dfRanked.sort_values('rate')

sumMaleRate = 0
sumFemaleRate = 0
rank = 1

for index, row in dfRankedRate.iterrows():
    if row["gender"] == 1:
        sumMaleRate += (rank + 1)
    elif row["gender"] == 2:
        sumFemaleRate += (rank + 1)
    else:
        print("Data at index {} is weird \n".format(index))
    rank += 1

print("Rank sum male: {}".format(sumMaleRate))
print("Rank sum female: {}".format(sumFemaleRate))
print("Fraction male sample vs female sample: {}".format(len(dfMale)/len(dfFemale)))
print("Normalized rank sum male: {}".format(sumMaleRate * 1/(sumMaleRate + sumFemaleRate) * 20*(20+1)/2))
print("Normalized rank sum female: {}".format(sumFemaleRate * 1/(sumMaleRate + sumFemaleRate) * 20*(20+1)/2))
print("Rank sum for males and females is not lower or higher than their bounds, Tl = 79, Tu = 131. Accept H0\n")


sns.set()

measurements = dfMale["temperature"]
stats.probplot(measurements, dist="norm", plot=pylab)
plt.title("QQ-plot for temperature")
pylab.show()

measurements = dfMale["rate"]
stats.probplot(measurements, dist="norm", plot=pylab)
plt.title("QQ-plot for heart rate")
pylab.show()