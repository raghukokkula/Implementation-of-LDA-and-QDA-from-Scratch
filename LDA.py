import numpy as np

import matplotlib.pyplot as plt


from sympy.solvers import solve
from sympy import Symbol
from sympy import poly

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+ 0.3*np.random.randn(200)

label= np.ones_like(x)

label[0:100]=0

plt.scatter(x,y,c=label)


x1,x2 = x[0:100], x[100:200]

y1,y2= y[0:100], y[100:200]


mean_x1  = np.mean(x1) 
mean_x2  = np.mean(x2)
mean_y1 = np.mean(y1)
mean_y2 = np.mean(y2)

mu1 = np.array([mean_x1, mean_y1])
#print("mu1",mu1)
mu2 = np.array([mean_x2, mean_y2])
#print("mu2 ",mu2)
sig_inv = np.linalg.inv(np.cov(x,y))
#print(sig_inv)

x1 = Symbol('x1')

x2 = Symbol('x2')

X = [x1,x2]


#X[0] = poly(X[0] - mu1[0])
#X[1] = poly(X[1] - mu1[1])


X_mu = [poly(X[0] - mu1[0]) , poly(X[1] - mu1[1]) ]
#print(X_mu)
#print("X ",np.shape(X),"\n")

#X = X - mu1
X_mu = np.reshape(X_mu, (1,2))
#print("X ",np.shape(X))
X_mu_transpose = np.transpose(X_mu)


X_lhs_sig = np.dot(X_mu, sig_inv)

X_lhs = np.dot(X_lhs_sig, X_mu_transpose)

#print("X ",X_lhs[0,0],"\n")

####---------- rhs



X_rhs = [x1,x2]


X_rhs_mu = [poly(X_rhs[0] - mu2[0]) , poly(X_rhs[1] - mu2[1]) ]
#print("X_rhs_mu",X_rhs_mu,"\n")

#X_r = X_r - mu2
X_rhs_mu = np.reshape(X_rhs_mu, (1,2))

X_rhs_mu_transpose = np.transpose(X_rhs_mu)

X_rhs_sig = np.dot(X_rhs_mu, sig_inv)

X_rhs = np.dot(X_rhs_sig, X_rhs_mu_transpose)

#print("X_rhs ",X_rhs,"\n")

equ = X_lhs - X_rhs

print("LDA equation ",equ)

x2_value = (solve(equ[0],x2))[x2].subs(x1,-1) 
x1_value = (solve(equ[0],x1))[x1].subs(x2,0)

a = [x1_value, -1]
b = [0,x2_value]

plt.plot(a,b)
plt.show




