# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:24:49 2022

@author: alons
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root

"""
Functions
"""

def labour(lt):
    return (lt**-α)-((1-φ)/φ)*(C_path[0]/((1-τ_l)*(1-α)*A*(K_path[0]**α)))-lt**(1-α)

def system(p):
    kt,ct,lt=p
    f0 = A*(K_path[0]**α)*(l_path[0]**(1-α))+(1-δ)*K_path[0]-C_path[0]-kt
    f1 = (lt**-α)-((1-φ)/φ)*(ct/((1-τ_l)*(1-α)*A*(kt**α)))-lt**(1-α)
    f2 = β*((1-τ_k)*α*A*(kt**(α-1))*(lt**(1-α))-δ+1)-(ct/C_path[0])*((((C_path[0]**φ)*((1-l_path[0])**(1-φ)))/((ct**φ)*((1-lt)**(1-φ))))**(1-σ))
    return f0,f1,f2

def system_shooting(m):
    kt,ct,lt=m
    g0 = A*(K_path[-1]**α)*(l_path[-1]**(1-α))+(1-δ)*K_path[-1]-C_path[-1]-kt
    g1 = (lt**-α)-((1-φ)/φ)*(ct/((1-τ_l)*(1-α)*A*(kt**α)))-lt**(1-α)
    g2 = β*((1-τ_k)*α*A*(kt**(α-1))*(lt**(1-α))-δ+1)-(ct/C_path[-1])*((((C_path[-1]**φ)*((1-l_path[-1])**(1-φ)))/((ct**φ)*((1-lt)**(1-φ))))**(1-σ))
    return g0,g1,g2

#Define the parameters of the model
β, γ, α, δ, φ, σ, A = 0.98, 2, 0.4, 0.08, 0.5, 2, 1
τ_k, τ_l = 0.4, 0.1


#Create the lists that are going to contain the values of the path for K_t, C_t and l_t
K_path = []
C_path = []
l_path = []

R = φ*(1-α)*A*((((1/β)+δ-1)*(1/((α*A)))))
G = φ*(1-τ_l)*(1-α)*A*((((1/β)+δ-1)*(1/((1-τ_k)*(α*A)))))

#Define the steady state values without taxes
l_ss = R/(R+((1-φ)*A*(((1/β)+δ-1)*(1/((α*A)))))-δ*(1-φ))
K_ss = l_ss*((((1/β)+δ-1)*(1/(α*A)))**(1/(α-1)))
C_ss = (φ/(1-φ))*(1-l_ss)*(1-α)*A*(K_ss**α)*(l_ss**-α)
    
#Define the steady state values with taxes
l_ss_tax = G/(G+((1-φ)*A*(((1/β)+δ-1)*(1/((1-τ_k)*(α*A)))))-δ*(1-φ))
K_ss_tax = l_ss_tax*((((1/β)+δ-1)*(1/((1-τ_k)*(α*A))))**(1/(α-1)))
C_ss_tax = (φ/(1-φ))*(1-l_ss_tax)*(1-τ_l)*(1-α)*A*(K_ss_tax**α)*(l_ss_tax**-α)

#Production levels in the steady state with and without taxes
Y_ss = A*(K_ss**(α))*(l_ss**(1-α))
Y_ss_tax = A*(K_ss_tax**(α))*(l_ss_tax**(1-α))
    
#Define a minimum and a maximum value for the 1st guess for c[0]
C_min = C_ss_tax
C_max = 2*C_ss
C_0a = (C_max+C_min)/2

while C_path==[]:
    #Define K[0] and l[0]
    K_path.append(K_ss)
    C_path.append(C_0a)
    l_path.append(float(fsolve(labour,0.5)))
      
    #Define K[1] and C[1]
    K_path.append((root(system_shooting,(K_path[0],C_path[0],l_path[0]),method='lm'))['x'][0])
    C_path.append((root(system_shooting,(K_path[0],C_path[0],l_path[0]),method='lm'))['x'][1])
    l_path.append((root(system_shooting,(K_path[0],C_path[0],l_path[0]),method='lm'))['x'][2])

    #Define an error
    err = 1e-4
    i = 0
    while K_path[i+1] < K_path[i] and C_path[i+1] < C_path[i]:
        K_path.append((root(system_shooting,(K_path[-1],C_path[-1],l_path[-1]),method='lm'))['x'][0])
        C_path.append((root(system_shooting,(K_path[-1],C_path[-1],l_path[-1]),method='lm'))['x'][1])
        l_path.append((root(system_shooting,(K_path[-1],C_path[-1],l_path[-1]),method='lm'))['x'][2])
        i += 1
    #Conditionals to check if we are in a steady state enviroment
    if abs(C_path[i+1]-C_ss_tax) < err:
        break
    #Conditional to check if our initials values are over or below the correct path
    elif C_path[i+1] > C_path[i]:
        C_path, K_path, l_path = [],[],[]
        C_max = C_0a
        C_0a = (C_min+C_max)/2
    elif K_path[i+1] > K_path[i]:
        C_path, K_path, l_path = [],[],[]
        C_min = C_0a
        C_0a = (C_min+C_max)/2  

"""
At the beginning with φ=0.5 the utility at the steady state is not positive. 
Hence, in order to have a positive utility in the steady state we have to change 
the value of φ to a value that provokes that individuals in the model valuate the amount of consumption.
"""

def utility(c,l):
    #For instance a value that works properly is φ = 0.95
    return ((((c**φ)*((1-l)**(1-φ)))**(1-σ))-1)/(1-σ)

U0 = utility(C_ss,l_ss)/(1-β) #Initial utility
U1 = utility(C_ss_tax,l_ss_tax)/(1-β) #Utility under taxes
gamma = 0.001
while U1 < U0:
    u1 = 0
    for i in range(len(C_path)):
        u1 = u1 + (β**i)*utility(C_path[i]*(1+gamma), l_path[i])
    U1 = u1 + utility(C_ss_tax*(1+gamma),l_ss_tax)/(1+β)
    gamma = gamma + 0.01
print(f'The welfare cost is {gamma}')