import numpy as np
import lib
import math
file = open('msfit.txt', 'r')
lines = file.readlines()
file.close()
T = []
N = []
E = []
for line in lines[1:]:
    p = np.fromstring(line, sep=' ')
    T.append(p[0])
    N.append(p[1])
    E.append(p[2])
E = np.array(E)
T = np.array(T)
N = np.array(N)
# print(1/E)

zz = lib.chi2(T, np.log(N), 1/E)
print("chi2 : ", zz[0])
print("chi2/dof : ", zz[0]/(len(T)-2))
print("Error in N_o : ", zz[6]*np.sqrt(zz[1]))
print("Error in lambda : ", math.sqrt(zz[2]))
print("covab : ", zz[3])
print("r2 : ", zz[4])
print("ln(N_0) : ", zz[5])
print("N_0 : ", math.exp(zz[5]))
print("b : ", zz[6])
print("lifetime (lambda): ", -zz[6])
