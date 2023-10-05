#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:54:20 2023

@author: celitona
"""

import numpy as np
from matplotlib import pyplot as plt
#from os import path

import ExactAlgorithm_WFb_estimation as ExWFest
import Unbiased_est_func as UnEst



from scipy.optimize import minimize_scalar
import csv

#%%
# =============================================================================
#Data generation and Monte Carlo unbiased estimation of h=\vartheta
##dXt=[{alpha(Xt)+eta(Xt, h)*Xt(1-Xt)}]dt+sqrt(Xt(1-Xt))dBt , 
#with alpha(Xt)=0.5*[theta1-(theta1+theta2)*Xt] and theta=theta1+theta2
##with eta(Xt)=h/2
#Functions:
# =============================================================================
def alpha ( x , theta):#Matrix of frequencies (ordered by loci) and matrix of 
#parent independent mutations.
    return 0.5* (theta[0] - (theta [0] + theta [1])*x[0])

def phi (x, theta, h):#Expects vector x with frequencies x at different sampled 
#times tj (one tj per each column) Vertix in abscisse: 0.5-sum(theta)
    return x[:]* (1-x[:]) * (h**2)/8 + h* 0.25* (theta[0] - (theta [0] + theta [1])*x[:])

def phi_plus (h, theta):
    #Works for small theta, like 0.01 or 0.001
    return (1/32)* h**2 + (1/8)*h*(theta[0]-theta[1]) + (1/8)* np.sum(theta)**2
                #, min \0* (1-0) * (h**2)/8 + h* 0.25* (theta[0] - (theta [0] + theta [1])*0))
    #for larger theta, say 0.9 the vertex is outside [0,1] and the max is at 0
    #return  theta[0]*h/4
    
def phi_minus (h, theta):
    return -theta[1]*h/4 #For D1
    #return theta[0]*h/4 #For D2
    #return min(-theta[1]*h/4, theta[0]*h/4)

def phi_max (h, theta):#For positive a=1/32, maximum should be at h= 1. For h=[-1, 1]
        #Remember that phi_max=\tilde(phi_max)+ max(-theta[1]*h/4, theta[0]*h/4)
        return max ((1/32)+ (1/8)*(theta[0]-theta[1]) + (1/8)* np.sum(theta)**2 +theta[0]/4,\
            (1/32)+ (1/8)*(theta[0]-theta[1]) + (1/8)* np.sum(theta)**2-theta[1]/4)
        #return theta[0]*h/4-theta[1]/4
        
def A (y, x, h):
    return (h/2) *(y-x)

def A_plus  (h):
    return np.abs(h/2)

#%%
# =============================================================================
# Constants for data generation
# =============================================================================

mut , h= 0.02, 0.7 #For D1
#mut , h= 0.1, -0.9 #For D2
init , loci , allele = 0.3, 1, 1
theta = mut * np.ones(allele+1)
t0, t = 0, 100
x0 = init * np.ones(loci*allele)

#%%
# =============================================================================
# Plots for testing
# ===============

var = np.linspace(-1, 1,num=30)
xes = np.linspace(0, 1,num=30)


plt.figure()
plt.axhline(y = phi_plus(h,theta), color = 'b', linestyle=':')#Blue
plt.plot(xes, phi(xes, theta, h), color = 'r')
plt.axhline(y = phi_minus(h,theta), color = 'g', linestyle=':')#Green
plt.show()


plt.figure()
plt.axhline(y = 50*phi_max(h, theta), color = 'k', linestyle=':')#Black
plt.axhline(y = phi_plus(h,theta)-phi_minus(h,theta), color = 'b', linestyle=':')#Blue
plt.plot(xes, phi(xes, theta, h) - phi_minus(h,theta), color = 'r')
plt.axhline(y = 0, color = 'g', linestyle=':')#Green
plt.show()


plt.figure()
plt.axhline(y = 1, color = 'k', linestyle=':')#Black
plt.plot(xes,(1/50*phi_max(h, theta))*(phi(xes, theta, h)- phi_minus(h,theta) ), color = 'r')
plt.axhline(y = 0, color = 'g', linestyle=':')#Green
plt.show()


plt.figure()
plt.axhline(y = 1, color = 'k', linestyle=':')#Black
plt.plot(xes,1- (1/10*phi_max(h, theta))*(phi(xes, theta, h)- phi_minus(h,theta) ), color = 'r')
plt.axhline(y = 0, color = 'g', linestyle=':')#Green
plt.show()


plt.figure()
plt.axhline(y = 1, color = 'b', linestyle=':')#Blue
plt.plot(xes,(1/(phi_plus(h,theta)-phi_minus(h,theta)))*(phi(xes, theta, h)- phi_minus(h,theta) ), color = 'r')
plt.axhline(y = 0, color = 'g', linestyle=':')#Green
plt.show()

plt.figure()
plt.axhline(y = A_plus  (h), color = 'k', linestyle=':')
plt.plot(xes, A (xes, 1, h) , color = 'r')
plt.show()

plt.figure()
plt.plot(var, phi_plus(var, theta) -phi_minus(var, theta), color = 'r')
plt.axhline(y = phi_max(var,theta), color = 'k', linestyle=':')#Black
plt.show()

#%%
# # =============================================================================
# # Data Generation:
# # =============================================================================
#Generation of 1 path
seed=241219 #For D1 and D2
np.random.seed(seed)
#Grid of time points: Bridge
n1=100
n=100
n2=n-n1
T_star1=np.append(np.arange(0, t,(t-0)/n1), t)

Bridge_mid1=np.zeros(n+1)
Bridge_mid1[0]=x0
for i in  range(1,n+1):
    Res=ExWFest.Exact_WFb_gen(Bridge_mid1[i-1],T_star1[i]-T_star1[i-1],phi_plus,phi_minus,phi, A, A_plus, theta, h)
    Bridge_mid1[i]=Res[1][-1]
    print("iteration", i, ", Samples", Res[1], "P. points", Res[3], "Attempts", Res[2], "\n" )
#Bridge1 = np.concatenate( ([Res[1][0]], Bridge_mid1, Z) )
#Bridge1 = np.concatenate( (Bridge_mid1, Z) )

Bridge1 = Bridge_mid1

T_star1
Bridge1

plt.figure()
plt.plot(T_star1, Bridge1, marker=".",linestyle = 'dotted')
plt.show()

T_star=T_star1
Bridge=Bridge1

plt.figure()
plt.plot(T_star, Bridge, marker=".",linestyle = 'dotted')
plt.show()


fig=plt.plot(T_star, Bridge, marker=".",linestyle = 'dotted', color='k')
plt.savefig('GeneratedPath.eps', format='eps')



#%% Number of MC samples
N=1000
#%%
#Generate and save MC samples of pM
# seed=241219 #For both D1 and D2
np.random.seed(seed)

with open(f'data/pM_{N}_{seed}_{h}_{theta}.csv', "w", encoding='utf-8' ) as f:#We print seed, N, number of contrib
    print(("{}\n""{}\n""{}\n").format(seed, N, len(Bridge)-1), end = "", file = f)
for j in range(N):
    kte=[]
    print("MC sample: ", j, "\n")
    for i in range(0,len(Bridge)-1):
        if(i % 10==0):
            # Print on the screen some message
            print("contribution: ", i, "\n")
        kte.append(UnEst.pM ( Bridge[i], Bridge[i+1], T_star[i+1]-T_star[i], theta))
    with open(f'data/pM_{N}_{seed}_{h}_{theta}.csv', "a", encoding='utf-8' ) as f:#We also print values of each contrib
        add_to_f=csv.writer(f)
        add_to_f.writerow(kte)

#%%
#Generate and save MC samples of a
#seed=241224 #For D1
#seed=131002 #For D2

np.random.seed(seed)

#if path.exists(f'data/a_{N}_{seed}_{h}_{theta}.csv', "w", encoding='utf-8' ): 
with open(f'data/a_{N}_{seed}_{h}_{theta}.csv', "w", encoding='utf-8' ) as f:#We print seed, N, number of contrib
    print(("{}\n""{}\n""{}\n").format(seed, N, len(Bridge)-1), end = "", file = f)
for j in range(N):
    print("MC sample: ", j, "\n")
    ome_list=[]
    for i in range(0,len(Bridge)-1):
        if(i % 10==0):
            # Print on the screen some message
            print("contribution: ", i, "\n")
        a=UnEst.a_est(Bridge[i], Bridge[i+1], T_star[i], T_star[i+1], theta, phi_max, phi_minus, phi, h)
        # convert a to a string:
        #  a=[] -> a_str=""
        #  a=[3] -> a_str="3"
        #  a=[1,2] -> a_str="1_2"
        a_str = ""
        if len(a) > 0:
            for a_j in a:
                a_str += "{}_".format(a_j)
            a_str = a_str[:-1]
        ome_list.append(a_str)
    with open(f'data/a_{N}_{seed}_{h}_{theta}.csv', "a", encoding='utf-8' ) as f:#We also print values of each contrib
        add_to_f=csv.writer(f)
        add_to_f.writerow(ome_list)

#%%
#Reading pM MC samples. They were stored as strings. Also, if they were logs, need to exp.
seed=241224 #For D1
#seed=131002 #For D2
pM_N = []
with open(f'data/pM_{N}_{seed}_{h}_{theta}.csv', "r", encoding='utf-8') as fp:
    for j, row in enumerate(csv.reader(fp)):
        if j < 3: #Remove 3 first lines!
            continue
        row_floats=[]
        for r in row:
            r_floats=[]
            r_floats = float(r)
            row_floats.append(r_floats)
        pM_N.append(row_floats)#Unpacking them per likelihood contributions

pM_N_T = []
pM_N_T=[[float(p_s) for p_s in p] for p in zip(*pM_N)]


#%%
#Reading a MC for all samples, and recovering K. They were stored as strings
seed=241224 #For D1
#seed=131002 #For D2
a_N, Ks = [], []
with open(f'data/a_{N}_{seed}_{h}_{theta}.csv', "r", encoding='utf-8') as fa:
    for j, row in enumerate(csv.reader(fa)):
        if j < 3: #Remove 3 first lines!
            continue
        # row is a list of strings
        # must convert it to list of lists of floats!
        # row = ["", "", "1_2", "4", ""] -> row_floats = [[], [], [1,2],[4],[]]
        row_floats, K = [], []
        for r in row:
            r_floats = []
            if len(r) > 0:
                r_str = r.split("_")
                r_floats = [float(r_s) for r_s in r_str]
            row_floats.append(r_floats)
            K.append(len(r_floats))
        a_N.append(row_floats)
        Ks.append(K)
        
ome, Ks_T = [], []

#Unpacking them per likelihood contributions
Ks_T=[[float(k_s) for k_s in k] for k in zip(*Ks)]
ome=[[a_s for a_s in a] for a in zip(*a_N)]

#%%
#Optimizing with Brent's method
seed=241219

params= (N,Bridge, T_star, theta, Ks_T, ome, A, phi, phi_max, phi_minus)#pM_N_T,

np.random.seed(seed)
Opt_tot_Brent=minimize_scalar(UnEst.min_logLikelihood,args=params, method='Brent')
print(Opt_tot_Brent.x) #MLE!

#Bootstrapped SE
np.random.seed(seed)
boots=50
MLE=[]
Ks_T_boot=[]
ome_boot=[]
for j in range(boots):
    if(j % 10==0):
        # Print on the screen some message
        print("Bootstrap sample: ", j, "\n")
    samples=np.random.choice(range(len(Ks_T[0])), size=N, replace=True, p=None)
    for i in range(len(Ks_T)):
        Ks_T_boot.append( list(np.take(Ks_T[i], samples)) )
        ome_boot.append( list(np.take(ome[i], samples) ))
    params_sample= (N,Bridge, T_star, theta, Ks_T_boot, ome_boot, A, phi, phi_max, phi_minus)
    Opt_tot_Brent=minimize_scalar(UnEst.min_logLikelihood,args=params_sample, method='Brent')
    MLE.append(Opt_tot_Brent.x)
    Ks_T_boot=[]
    ome_boot=[]
np.std(MLE)

#%%
#Generate MC samples of the full loglikelihood
##Extracting n samples a (, i.e., KS and ome), for n=10, 20, 50, 100, 200, 500
seed=241219 #For D1 and D2
np.random.seed(seed)
n10=10;n20=20;n50=50;n100=100;n200=200;n500=500;n1=1;
n=[n10,n20,n50,n100, n200,n500, n1]
index=[]
for j in n:
    index.append(np.random.choice(range(N), size=j, replace=False, p=None))

Ks_T_sample=[]
ome_sample=[]
ind=5#loop over all of n's entries 0, 1, 2, 3, 4, 5,6
index[ind]
n[ind]

for j in range(len(Ks_T)):
    Ks_T_sample.append( list(np.take(Ks_T[j], index[ind])) )
    ome_sample.append( list(np.take(ome[j], index[ind]) ))


params_sample= (n[ind],Bridge, T_star, theta, Ks_T_sample, ome_sample, A, phi, phi_max, phi_minus)
ToPlot=list()
num = np.linspace(-1, 1,num=50).tolist()
for par in num:
    ToPlot.append(-UnEst.min_logLikelihood (par,*params_sample))

plt.figure()
plt.plot(num, ToPlot)
plt.show()

num[np.argmax(ToPlot)]

np.random.seed(seed)
Opt_tot_Brent=minimize_scalar(UnEst.min_logLikelihood,args=params_sample, method='Brent')
print(Opt_tot_Brent.x)

# #For N=1: -h=0.7, theta=0.02, 0.02
# MLE11=np.array([1.8663529695820191,1.8547101343378056, 0.6964547152564636, 0.2566425301164854, 1.0354408317560608,  
#                 2.9176899014104802,1.221377192723868, 1.3349667841536912, 1.9101943819262346 ,0.26013927971376904,
#                 0.8790900210099168, 0.8690947001976482 ] )
# np.std(MLE11)

# #For N=1: -h=-0.9, theta=0.1, 0.1
# MLE12=np.array([-1.2791489825227824,-1.3618967887889775,-0.5280927705592353, -1.6168343395157496, -0.8653087474292892, 
#           -1.0713783185764543,-0.7704520740102605, -0.598010476442279, -1.0356073955496676, -1.1994442717732654,
#           -0.8482687811447555, -1.1278886701588353, -0.6993634426709365, -1.540078443428925 ,-0.3424470723974676,
#           -0.8699149216197478, -0.7276141100685228, -1.2738186563912015,  -0.6866566184955301, -1.097509840783731 ] )
# np.std(MLE12)

#Bootstrapped SE
np.random.seed(seed)
boots=50
MLE=[]
Ks_T_boot=[]
ome_boot=[]
for j in range(boots):
    if(j % 10==0):
        # Print on the screen some message
        print("Bootstrap sample: ", j, "\n")
    samples=np.random.choice(range(len(Ks_T_sample[0])), size=n[ind], replace=True, p=None)
    for i in range(len(Ks_T)):
        Ks_T_boot.append( list(np.take(Ks_T_sample[i], samples)) )
        ome_boot.append( list(np.take(ome_sample[i], samples) ))
    params_sample= (n[ind],Bridge, T_star, theta, Ks_T_boot, ome_boot, A, phi, phi_max, phi_minus)
    Opt_tot_Brent=minimize_scalar(UnEst.min_logLikelihood,args=params_sample, method='Brent')
    MLE.append(Opt_tot_Brent.x)
    Ks_T_boot=[]
    ome_boot=[]
np.std(MLE)