# .py-file for testing various syntax from different libs that involves
# simple statistics

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from itertools import imap

vals = np.random.normal(0, 0.5, 10000)

# plt.hist(vals, 50)
# plt.show()

statsDict = dict()

statsDict['myVals'] = np.mean(vals)
statsDict['varVals'] = np.var(vals)
statsDict['skewVals'] = sp.skew(vals)
statsDict['kurtVals'] = sp.kurtosis(vals)

for key in statsDict.keys():
    lenStr = len(key)
    print(lenStr)

#print('-'*8 + '|-------')
#for key in statsDict.keys():
#print(key + '|' + str(statsDict[key]))
