
import numpy as np

# import library
import lib
# open matrix
with open('mstrimat.txt') as f:
    a = []
    for i in range(0, 5):
        a.append(list(map(float, f.readline().split())))
A = np.array(a)
# print(a)

# first eigenvalue and vector
xs1 = lib.power_method(A)
print("First eigenvalue : ", xs1[0], "and corresponding Eigenvector : ", xs1[1])

# second eigenvalue and vector
Aa = A-xs1[0]*np.outer(xs1[1], xs1[1])
# print(Aa)
xs2 = lib.power_method(Aa)
print("First eigenvalue : ", xs2[0], "and corresponding Eigenvector : ", xs2[1])

# cross verifiaction
a = -1
c = -1
b = 2
n = 5
k = np.array([1, 2, 3, 4, 5])
i = np.array([1, 2, 3, 4, 5])

lk = b+2*np.sqrt(a*c)*np.cos(k*np.pi/(n+1))
print("Eigenvalues", lk)

for j in range(1, 6):
    vk = 2*np.power(np.sqrt(c/a), j)*np.sin(i*j*np.pi/(n+1))
    print(j, " Eigen vector :", vk)
