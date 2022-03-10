import numpy as np
# import library
import lib
# open a and b
with open('A.txt') as f:
    a = []
    for i in range(0, 6):
        a.append(list(map(float, f.readline().split())))
a = np.array(a)
opn = open('B.txt', 'r')
lsplit = opn.readline().split()
b = []
for val in lsplit:
    b.append(float(val))
b = np.array(b)

xs = lib.gauss_seidel(a, b, 1e-5)
print('\nSolution by Gauss Seidel: x = {}'.format(xs))


# jacobieq

xs2 = lib.jacobieq(a, b, 1e-5)
print('\nSolution by Jacobi : x = {}'.format(xs2))
