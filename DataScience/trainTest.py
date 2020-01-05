import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds

sns.set()

# 80/20 - split of the data
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]


plt.scatter(trainX, trainY)
plt.title('Training Data')
plt.show()

plt.scatter(testX, testY)
plt.title('Test Data')
plt.show()

x = np.array(trainX)
y = np.array(trainY)
p4 = np.poly1d(np.polyfit(x, y, 5))  # constuct function based on fitted v.

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.show()

# where testy are the values p4(testx) is trying to predict
r2 = r2_score(testy, p4(testx))
print(r2)
