# testing different plot options in matplotlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.arange(-3, 3, 0.001)


# Classic line plot

axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks(np.linspace(-5, 5, 11))
axes.set_yticks(np.linspace(0.0, 1.0, 11))
axes.grid()
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1, 0.5))
plt.show()


# Pie chart

values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]
labels = ['India', 'United States', 'Russia', 'China', 'Europe']
plt.pie(values, explode, labels, colors)
plt.title('Student Locatations')
plt.show()


# Bar chart

plt.bar(range(1, 6), values, color=colors, tick_label=labels)
plt.show()


# Box and Whisker plot

uniformSkewed = np.random.rand(100)*100 - 40
high_outliers = np.random.rand(10)*50 + 100
low_outliers = np.random.rand(10)*(-50) - 100
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
plt.boxplot(data)
plt.show()
