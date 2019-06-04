import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.6, 0.1],
              [0.5, 0.2, 0.3],
              [0.4, 0.1, 0.5]])

v = np.ones(3)/3
s = []

for i in range(25):
    v_prim = v.dot(A)
    s.append(np.linalg.norm(v_prim - v))
    v = v_prim

plt.plot(s)
plt.show()
