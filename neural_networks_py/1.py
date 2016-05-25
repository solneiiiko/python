# import numpy as np
# y = [1, 0, 0]
# a = [1, 0.3, 0.1]

# n = len(y)

# sq = 0
# ce = 0
# for i in range(n):
#   sq += (y[i]-a[i])**2 
#   l = 1-a[i]
#   if l==0: l = np.e
#   ce += (y[i]*np.log(a[i])) + (1-y[i])*np.log(l)
# sq *= 1/n
# ce *= -1/n
# print('SQ: ' + str(sq))
# print('CE ' + str(ce))

# def C(n,k):
#     if k==0:
#         return 1
#     elif k>n:
#         return 0 
#     else:
#         return C(n-1,k)+C(n-1,k-1)
import math
def C(n,k):
    f = math.factorial
    return f(n)/(f(k)*f(n-k))
n = 50
p = 0.55
q = 0.45
sums = 0
for k in range(26,51):
    P = C(n,k) * (p**k) * (q**(n-k))
    sums += P
    print("k = "+str(k)+' : '+str(P))
k = 25
P = C(n,k) * (p**k) * (q**(n-k))
sums += P/2

print('ИТОГ: '+str(sums))