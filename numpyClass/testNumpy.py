import numpy as np

L = [1, 2, 3]
A = np.array([1, 2, 3])

L.append(4)     # doesn't work on numpyArray
L = L + [5]

L2 = []
for e in L:
    L2.append(e + e)

A2 = A + A   # You can do this instead of looping if you are using an numpyArray

2*A     # Scalar multiplication
LL = 2*L     # Makes an identical array after the original one

Lp = []
for e in L:
    Lp.append(e*e)

Ap = A**2   # Same shit, a lot easier

Asq = np.sqrt(A)
Alog = np.log(A)
Aexp = np.exp(A)

