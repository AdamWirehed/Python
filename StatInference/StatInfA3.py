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

print("Mean xyBar: {}\n".format(xyBar))
print("SE xy: {}".format(SE_xy))
print("Confidence interval (95%, normal): {}\n".format(I_normal))

# task ii - parametric test (assuming normal dist.) = two sample t-test

totSize = len(dfMale) + len(dfFemale)
s2Pooled = (len(dfMale) - 1)/(totSize - 2)*s2Male + (len(dfFemale) - 1)/(totSize - 2)*s2Female
degF = totSize - 2
t_obs = xyBar/m.sqrt(s2Pooled) * m.sqrt((len(dfMale) * len(dfFemale))/totSize)

print("t-value: {}".format(t_obs))
print("Df: {}".format(degF))
print("p-value: {}\n".format(2*(1 - stats.t.cdf(abs(t_obs), degF))))
print(stats.ttest_ind(dfMale["temperature"], dfFemale["temperature"]))


# task iii, Rank sum test

print(stats.ranksums(dfMale["temperature"], dfFemale["temperature"]))

# task b, i

xyBarRate = dfMale["rate"].mean() - dfFemale["rate"].mean()

s2MaleRate = dfMale["rate"].var()
s2FemaleRate = dfFemale["rate"].var()

SE_xyRate = m.sqrt(s2MaleRate/len(dfMale) + s2FemaleRate/len(dfFemale))
I_normalRate = [xyBarRate - 1.96*SE_xyRate, xyBarRate + 1.96*SE_xyRate]

print("Mean xyBar: {}".format(xyBarRate))
print("SE xy: {}".format(SE_xyRate))
print("Confidence interval (95%, normal): {}\n".format(I_normalRate))

# b,ii - parametric test, two sample t-test

totSize = len(dfMale) + len(dfFemale)
s2PooledRate = (len(dfMale) - 1)/(totSize - 2)*s2MaleRate + (len(dfFemale) - 1)/(totSize - 2)*s2FemaleRate
degF = totSize - 2
t_obsRate = xyBarRate/m.sqrt(s2PooledRate) * m.sqrt((len(dfMale) * len(dfFemale))/totSize)

print("t-value: {}".format(t_obsRate))
print("Df: {}".format(degF))
print("p-value: {}\n".format(2*(1 - stats.t.cdf(abs(t_obsRate), degF))))
print(stats.ttest_ind(dfMale["rate"], dfFemale["rate"]))

print(stats.ranksums(dfMale["rate"], dfFemale["rate"]))

sns.set()

measurements = dfMale["temperature"]
stats.probplot(measurements, dist="norm", plot=pylab)
plt.title("QQ-plot for temperature")
pylab.show()

measurements = dfMale["rate"]
stats.probplot(measurements, dist="norm", plot=pylab)
plt.title("QQ-plot for heart rate")
pylab.show()