import numpy as np
import scipy.stats as sc
import scipy.special as sp
from scipy import optimize as opt
import math as m
import matplotlib.pyplot as plt
import seaborn as sns

filepath = "ASCII_Comma/Chapter_8/gamma-arrivals.txt"
data = []

file = open(filepath, 'r')

for line in file:
    data.append(float(line))

file.close()
data = np.asarray(data)

# Task b
xBar = data.mean()
x2Bar = pow(data, 2).mean()
alphaBar = xBar*xBar/(x2Bar - xBar*xBar)
lamBar = alphaBar/xBar
print("AlphaBar: {} \nGamBar: {} \n".format(alphaBar, lamBar))

gamParam = sc.gamma.fit(data, floc=0)
alphaHat = gamParam[0]
lamHat = 1/gamParam[2]   # Since fit returns scale instead of lambda

print("AlphaHat: {} \nLambdaHat: {} \n".format(alphaHat, lamHat))

# Task c
sampN = 100
alphaSamp = np.zeros([sampN, 1])
lamSamp = np.zeros([sampN, 1])
for ix in range(0, sampN):
    gamSample = sc.gamma.rvs(a=alphaHat, scale=1/lamHat, size=len(data), random_state=ix)
    xBarSamp = gamSample.mean()
    x2BarSamp = pow(gamSample, 2).mean()
    alphaSamp[ix] = xBarSamp*xBarSamp/(x2BarSamp - xBarSamp*xBarSamp)
    lamSamp[ix] = alphaSamp[ix]/xBarSamp

aBarSamp = alphaSamp.mean()
lamBarSamp = lamSamp.mean()

saHat = m.sqrt(1/sampN*((alphaSamp - aBarSamp)*(alphaSamp - aBarSamp)).sum())
slamHat = m.sqrt(1/sampN*((lamSamp - lamBarSamp)*(lamSamp - lamBarSamp)).sum())

print("Standard error for alpha hat: {} \nStandard error for lambda hat: {} \n".format(saHat, slamHat))

print("Alpha bar from est. gamDist: {} \nLambda bar from est. gamDist: {} \n".format(aBarSamp, lamBarSamp))

sns.set()

# a
# Seems to follow the gamma distrubution fairly well where alpha=1 (shape parameter)
plt.hist(data, bins=np.arange(min(data), max(data) + 10, 10), density=True, label="data")
plt.plot(sorted(data), sc.gamma.pdf(sorted(data), a=alphaBar, scale=1/lamBar), '-r', label="Gamma dist. MME (a=~1, gam=0.1267)")
plt.title("Interarrivals times")
plt.xlabel("time [s] (bin width = 10)")
plt.ylabel("Occurances")
plt.legend()
plt.show()

plt.hist(gamSample, bins=np.arange(min(data), max(data) + 10, 10), density=True, label="data")
plt.plot(sorted(gamSample), sc.gamma.pdf(sorted(gamSample), a=alphaBar, scale=1/lamBar), '-r', label="Gamma dist. MME (a=~1, gam=0.1267)")
plt.title("Interarrivals times, sampled from estimated Gamma dist.")
plt.xlabel("time [s] (bin width = 10)")
plt.ylabel("Occurances")
plt.legend()
plt.show()
