import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

s1 = np.random.normal(loc=0, scale=1, size=6800)
s4 = np.random.normal(loc=0, scale=1, size=40) 

shift = 0.00001
s2 = np.random.normal(loc=shift, scale=1, size=6800)
s3 = np.random.normal(loc=shift, scale=1, size=40)

res1 = ks_2samp(data1=s1, data2=s2)
print(f'{res1} \n')

res2 = ks_2samp(data1=s1, data2=s3)
print(f'{res2} \n')

res3 = ks_2samp(data1=s4, data2=s3)
print(f'{res3} \n')

sns.set()

fig1 = plt.figure()
plt.hist(s1, len(s1), density=True, histtype='step', cumulative=True, label=f'Empirical s1 (n={len(s1)})')
plt.hist(s2, len(s1), density=True, histtype='step', cumulative=True, label=f'Empirical s2 (n={len(s2)}), mean={shift}')
plt.legend()
plt.title("Empirical cumulative distribution function")

fig2 = plt.figure()
plt.hist(s1, len(s1), density=True, histtype='step', cumulative=True, label=f'Empirical s1 (n={len(s1)})')
plt.hist(s3, len(s3), density=True, histtype='step', cumulative=True, label=f'Empirical s3 (n={len(s3)}), mean={shift}')
plt.legend()
plt.title("Empirical cumulative distribution function")

fig3 = plt.figure()
plt.hist(s1, len(s1), density=True, histtype='step', cumulative=True, label=f'Empirical s1 (n={len(s1)})')
plt.hist(s4, len(s4), density=True, histtype='step', cumulative=True, label=f'Empirical s4 (n={len(s4)}), mean=0')
plt.legend()
plt.title("Empirical cumulative distribution function")

fig4 = plt.figure()
plt.hist(s1, len(s3), density=True, histtype='step', cumulative=True, label=f'Empirical s3 (n={len(s3)})')
plt.hist(s4, len(s4), density=True, histtype='step', cumulative=True, label=f'Empirical s4 (n={len(s4)}), mean=0')
plt.legend()
plt.title("Empirical cumulative distribution function")

plt.show()