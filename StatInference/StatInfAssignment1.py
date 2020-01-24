import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math as m

# Create dataframe from data
filepath = "ASCII_Comma/Chapter_7/families.txt"
df = pd.read_csv(filepath)

# Random sample without replacement and same "seed" each time
sampN = 600
sampleOne = df.sample(n=sampN, random_state=1)

# Standard error and confidence interval for Husband-Wife
sampleHW = sampleOne["'TYPE'"].value_counts()
pHW = sampleHW.get(1)/(sampleHW.values.sum())
sHW_p = m.sqrt(pHW * (1 - pHW)/(sampN - 1))
I_HW_p = [pHW - 1.96*sHW_p, pHW + 1.96*sHW_p]

print("Husband-wife \n-------------- \nProportion: {} \nStandard Error: {} \nConfidence Interval (95%): {}\n".format(pHW,
    sHW_p, I_HW_p))


# Standard error and confidence interval for nr of children in a family
xBarChild = sampleOne["'CHILDREN'"].mean()
stdChild = sampleOne["'CHILDREN'"].std()    # N-1 default (correct this is a sample)
sChild_xBar = stdChild/m.sqrt(sampN - 1)
I_Child_xBar = [xBarChild - 1.96*sChild_xBar, xBarChild + 1.96*sChild_xBar]

print("Nr of Children per family \n-------------- \nMean: {} \nStandard Error: {} \nConfidence Interval (95%): {}\n".format(
    xBarChild, sChild_xBar, I_Child_xBar))


# Standard error and confidence interval for nr of people in a family
xBarPeople = sampleOne["'PERSONS'"].mean()
stdPeople = sampleOne["'PERSONS'"].std()
sPeople_xBar = stdPeople/m.sqrt(sampN - 1)
I_People_xBar = [xBarPeople - 1.96*sPeople_xBar, xBarPeople + 1.96*sPeople_xBar]

print("Nr of persons per family \n-------------- \nMean: {} \nStandard Error: {} \nConfidence Interval (95%): {}\n".format(
    xBarPeople, sPeople_xBar, I_People_xBar))


# Task b

sampSeed = 0
sampN2 = 400
it = 100
nuPop = df["'INCOME'"].mean()
print("Population mean income: {}".format(nuPop))
xBarVec = np.zeros([it, 1])
stdVec = np.zeros([it, 1])
I_Vec = np.zeros([it, 2])
I_hitRate = np.zeros([it, 1])

for ix in range(0, it):
    samp = df.sample(n=sampN2, random_state=sampSeed)
    xBarVec[ix] = samp["'INCOME'"].mean()
    stdVec[ix] = samp["'INCOME'"].std()
    s_xBar = stdVec[ix]/m.sqrt(sampN2) * m.sqrt(1 - sampN2 / len(df))   # Since we finite population
    I_Vec[ix, :] = [xBarVec[ix] - 1.96*s_xBar, xBarVec[ix] + 1.96*s_xBar]
    if (I_Vec[ix, 0] <= nuPop) and (nuPop <= I_Vec[ix, 1]):
        I_hitRate[ix] = 1
    sampSeed += 1

print("Nr of confidence containing pop. income mean: {}".format(int(I_hitRate.sum())))

sampN3 = 100
xBar100 = np.zeros([it, 1])
std100 = np.zeros([it, 1])
sampSeed100 = 200

for ix in range(0, it):
    samp100 = df.sample(n=sampN3, random_state=sampSeed100)
    xBar100[ix] = samp100["'INCOME'"].mean()
    std100[ix] = samp100["'INCOME'"].std()
    sampSeed100 += 1

# Task c
dfNorth = df[df["'REGION'"] == 1]
dfEast = df[df["'REGION'"] == 2]
dfSouth = df[df["'REGION'"] == 3]
dfWest = df[df["'REGION'"] == 4]

pNorth = len(dfNorth)/len(df)
pEast = len(dfEast)/len(df)
pSouth = len(dfSouth)/len(df)
pWest = len(dfWest)/len(df)

nWinds = 500

sampNorth = dfNorth.sample(round(pNorth*nWinds), random_state=0)
sampEast = dfEast.sample(round(pEast*nWinds), random_state=1)
sampSouth = dfSouth.sample(round(pSouth*nWinds), random_state=2)
sampWest = dfWest.sample(round(pWest*nWinds), random_state=3)

sampWinds = pd.concat([sampNorth, sampEast, sampSouth, sampWest])
xBarWinds = sampWinds["'INCOME'"].mean()

print(xBarWinds)


# Ploting

sns.set()

plt.hist(xBarVec, density=True)
plt.title("Mean income in familes of 100 samples, n=400")
plt.xlabel("Income")
plt.ylabel("Observations")
plt.plot(sorted(xBarVec), sp.norm.pdf(sorted(xBarVec), xBarVec.mean(), xBarVec.std()), '-r')
plt.show()

plt.hist(xBar100, density=True)
plt.title("Mean income in familes of 100 samples, n=100")
plt.xlabel("Income")
plt.ylabel("Observations")
plt.plot(sorted(xBar100), sp.norm.pdf(sorted(xBar100), xBar100.mean(), xBar100.std()), '-r')
plt.show()

