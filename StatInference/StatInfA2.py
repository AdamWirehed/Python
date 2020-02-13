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
aMME = np.zeros([sampN, 1])
lamMME = np.zeros([sampN, 1])
aMLE = np.zeros([sampN, 1])
lamMLE = np.zeros([sampN, 1])

# Doing 100 simulated samples from a Gamma(alpha, lambda) dist based on the estimated parameters from MME and MLE
for ix in range(0, sampN):
    #Sampling from the different estimated gamma dist.
    gSampMME = sc.gamma.rvs(a=alphaBar, scale=1/lamBar, size=len(data), random_state=ix)
    gSampMLE = sc.gamma.rvs(a=alphaHat, scale=1/lamHat, size=len(data), random_state=ix)

    xBarMME = gSampMME.mean()
    x2BarMME = pow(gSampMME, 2).mean()
    aMME[ix] = xBarMME*xBarMME/(x2BarMME - xBarMME*xBarMME)
    lamMME[ix] = aMME[ix]/xBarMME

    xBarMLE = gSampMLE.mean()
    x2BarMLE = pow(gSampMLE, 2).mean()
    aMLE[ix] = xBarMLE * xBarMLE / (x2BarMLE - xBarMLE * xBarMLE)
    lamMLE[ix] = aMLE[ix] / xBarMLE


aBarMME = aMME.mean()
lamBarMME = lamMME.mean()
aBarMLE = aMLE.mean()
lamBarMLE = lamMLE.mean()


saHatMME = m.sqrt(1/sampN*((aMME - aBarMME)*(aMME - aBarMME)).sum())
slamHatMME = m.sqrt(1/sampN*((lamMME - lamBarMME)*(lamMME - lamBarMME)).sum())
saHatMLE = m.sqrt(1/sampN*((aMLE - aBarMLE)*(aMLE - aBarMLE)).sum())
slamHatMLE = m.sqrt(1/sampN*((lamMLE - lamBarMLE)*(lamMLE - lamBarMLE)).sum())

print("Standard error for alpha hat(MME): {} \nStandard error for lambda hat(MME): {} \n".format(saHatMME, slamHatMME))
print("Standard error for alpha hat(MLE): {} \nStandard error for lambda hat(MLE): {} \n".format(saHatMLE, slamHatMLE))


#d

I_saMME = [alphaBar - 1.96*saHatMME, alphaBar + 1.96*saHatMME]
I_sLamMME = [lamBar - 1.96*slamHatMME, lamBarMME + 1.96*slamHatMME]

I_saMLE = [alphaHat - 1.96*saHatMLE, alphaHat + 1.96*saHatMLE]
I_sLamMLE = [lamHat - 1.96*slamHatMLE, lamHat + 1.96*slamHatMLE]

print("Confidence interval (95%) alpha MME: {} \n".format(I_saMME))
print("Confidence interval (95%) lambda MME: {} \n".format(I_sLamMME))
print("Confidence interval (95%) alpha MLE: {} \n".format(I_saMLE))
print("Confidence interval (95%) lambda MLE: {} \n".format(I_sLamMLE))

sns.set()

# a
# Seems to follow the gamma distrubution fairly well where alpha=1 (shape parameter)
plt.hist(data, bins=np.arange(min(data), max(data) + 10, 10), density=True, label="data")
plt.plot(sorted(data), sc.gamma.pdf(sorted(data), a=alphaBar, scale=1/lamBar), '-r', label="Gamma dist. MME (a={:.4}, gam={:.4})".format(alphaBar, lamBar))
plt.plot(sorted(data), sc.gamma.pdf(sorted(data), a=alphaHat, scale=1/lamHat), '-g', label="Gamma dist. MLE (a={:.4}, gam={:.4})".format(alphaHat, lamHat))
plt.title("Interarrivals times")
plt.xlabel("time [s] (bin width = 10)")
plt.ylabel("Occurances")
plt.legend()
plt.show()

