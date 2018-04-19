#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:30:57 2018

@author: askeladd
"""

import numpy as np

import matplotlib.pyplot as plt


from sympy.solvers import solve
from sympy import *

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
sig_inv1 = np.linalg.inv(np.cov(x1,y1))
sig_inv2 = np.linalg.inv(np.cov(x2,y2))
#print(sig_inv1)
#print(sig_inv2)

x1 = Symbol('x1')

x2 = Symbol('x2')

X = [x1,x2]


qda_X_mu = [poly(X[0] - mu1[0]) , poly(X[1] - mu1[1]) ]
#print(qda_X_mu)
#print("X ",np.shape(qda_X_mu),"\n")


qda_X_mu = np.reshape(qda_X_mu, (1,2))
#print("X ",np.shape(qda_X_mu))
qda_X_mu_transpose = np.transpose(qda_X_mu)


qda_X_lhs_sig = np.dot(qda_X_mu[0], sig_inv1)

qda_X_lhs = np.dot(qda_X_lhs_sig, qda_X_mu_transpose)





qda_X_rhs = [x1,x2]


qda_X_rhs_mu = [poly(qda_X_rhs[0] - mu2[0]) , poly(qda_X_rhs[1] - mu2[1]) ]
#print("X_rhs_mu",qda_X_rhs_mu,"\n")

#X_r = X_r - mu2
qda_X_rhs_mu = np.reshape(qda_X_rhs_mu, (1,2))

qda_X_rhs_mu_transpose = np.transpose(qda_X_rhs_mu)

qda_X_rhs_sig = np.dot(qda_X_rhs_mu[0], sig_inv2)

qda_X_rhs = np.dot(qda_X_rhs_sig, qda_X_rhs_mu_transpose)

#print("X_rhs ",qda_X_rhs,"\n")

qda_equ = qda_X_lhs - qda_X_rhs
qda_equ = qda_equ[0]
print("QDA equation ",qda_equ)


a = np.linspace(-3,3,50)
b = np.linspace(-3,3,50)


mesh_grid_x, mesh_grid_y = np.meshgrid(a,b)

plot_array = [None]*50

for i in range(0,50):
    
    qda_plot = [None]*50
    
    for j in range(0,50):

        qda_plot[j] = (qda_equ.subs({x1:mesh_grid_x[i][j], x2:mesh_grid_y[i][j]}))
           
    plot_array[i] = ( qda_plot )

    
    


plt.contour(mesh_grid_x, mesh_grid_y, plot_array,1)
plt.show()
       
























