#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:25:31 2023

@author: celitona
"""
import numpy as np

import ExactAlgorithm_WFb_estimation as ExWFest
from Auxiliary_EWF import EWF_gen #For neutral WFB candidates
import Auxiliary_NeutralWF as ANWF

from scipy.stats import beta
from scipy.stats import binom

import warnings
import math


#import csv

# =============================================================================
# Unbiased estimator of the likelihood for two consecutive data points x and y
#occurring at a distance t
##L(\Lambda, h)=exp(A(y;h)-A(x,h)-tphi_minus)*p(x,y;t)*a(x,y;h) with
##p(x,y;t)=E_m[E_L[Beta()]]
##a(x,y,h)=\prod_1^K (1-phi (x, theta, h)-phi_minus)/phi_max
##K=number of sampled Poisson points
#with MC estimators
##pN( x , y, theta) #Theta is a vector of mutation rates
##aN(x,y,h)
# =============================================================================
def pM ( x , y, t, theta):#Monte Carlo sample of neutral cond WFdiff
    #sum1=0
    #for _ in range(N):
    #m = ANWF.death_process( np.sum(theta), ANWF.b, ANWF.a, ANWF.S_minus,
                        #ANWF.S_plus, t, ANWF.griffiths_approx)
    m=ANWF.griffiths_approx( np.sum(theta), t)
    m=np.array([m])
    #Samples stored in m[0]
    print("m=", m, "x=", x, "y=", y,  "\n")
    sum0 = 0
    for l in range(m[0]+1):
        sum0+=binom.pmf(l, m[0], x)*beta.pdf(y, theta[0]+l, theta[1]+m[0]-l)
    if(math.isinf(sum0)==True):
        sum0=3000000
    return sum0

#a_est for single contribution from x at tx to y at ty using EWF
def a_est (x, y, tx, ty,theta, phi_max, phi_minus, phi,h):
    #K=0
    #ome=np.zeros((0,K))
    flag=0
    while flag != 1:
        lmbda = 20*phi_max(h, theta)
        area = (ty-tx)*(1-0)
        K = np.random.poisson(lmbda*area)
        #print("K,first =", K,"\n")
        # ome=np.zeros((0,K))
        if (K==0):
            # return ome
            return np.zeros(0)
        else:
            pts = np.random.uniform(tx, ty, K)
            xpts=np.random.uniform(0, 1, K)
            tpoints = pts[np.argsort(pts)]
            xpoints=xpts[np.argsort(xpts)]
            #print("tpoints=", tpoints, "xpoints=", xpoints, "K=", K, "\n" )
            ome=np.reshape(EWF_gen(tpoints, tx, ty, x, y, theta), K)
            # print("ome=", ome, "\n")
            flag = [1 if np.all( ( (phi(ome,theta,h) - phi_minus(h, theta))/ (20*phi_max (h, theta) ) ) <=xpoints) else 0] #4.1) Rejection scheme
            #print("Flag", flag, "Ys", xpoints, "K", K, "\n")
            if flag==[1]:
                #print("K", K,"ome", ome, "\n")
                return ome

#WFor the logLikelihood, we need to compute the terms:
###with a(jx^i,y^i,t^i)= prod_K 1- (phi(ome,h)-phi^-(h))/(phi^+-phi^-(h))
#N represents the MC samples that we want to use, and the length of the vectors p and a
##We will compute the product for each i, and then sum them all together
#K is a list of length N, containing MC samples for the K sampled points (columns of files)
#ome is a list contains arrays with sampled accepted candidates
#outputs the MC average over N samples of the product in (*)
def L_cont_a (N, K, ome,theta,h, phi, phi_max, phi_minus):#we only need to average over p and a
    a=np.zeros(N)
    pos=np.where(np.array(K)!=0)[0]
    for j in pos:
        B= 1- (1/(20*phi_max(h, theta)))*(phi(np.array(ome[j]), theta, h)- phi_minus(h,theta) )
        #B=1- ( ( 1/ phi_max(h, theta) ) * \
            #( (h**2/8)* np.array(ome[i])[:]* (1- np.array(ome[i])[:] ) + \
            #0.25*h * sum(theta) - \
            #0.25* sum(theta) * h* np.array(ome[i])[:] )  )
        #print("ome=", np.array(ome[j]),"\n")
        #print("B=", B, "\n")
        a[j]=np.prod(B)
        #print("a[j]=", a[j],"j=", j,"\n")
    #C= (1/N)*(sum(pMl*a))
    #print("a=", a,"\n")
    C= (1/N)*(sum(a))
    #print("C=", C,"\n")
    return C

#pM is a list of size  (len(Bridge)-1) x n, containing  n MC samples for the contribution from x to y (columns of files)
#K is a list of size (len(Bridge)-1) x n, containing n MC samples for the K sampled points (columns of files)
#ome is a list of size (len(Bridge)-1) x n containing n MC arrays of size K with sampled WFB candidates
def min_likelihood (h,*params):
    N,Bridge, T_star, theta, K, ome, A , phi, phi_max, phi_minus= params #pMl, 
    MC_conts=np.zeros(len(Bridge)-1)#where len(Bridge)-1 is the number of contributions
    conts=np.ones(len(Bridge)-1)
    for i in range(len(Bridge)-1):
        MC_conts[i]= L_cont_a(N, K[i],ome[i],theta,h, phi, phi_max, phi_minus)
        if (MC_conts[i]==0.):
            conts[i]=1
        else:
            conts[i]= np.exp( A( Bridge[i+1], Bridge[i] , h) - \
                 (T_star[i+1]-T_star[i])*phi_minus(h, theta) ) * MC_conts[i]#*np.mean(pMl[i]) #np.mean(pMl[i]) *
    #print("MC_conts=", MC_conts, "\n")
    Likelihood=(-1)*np.prod(conts)
    return Likelihood

#pM is a list of size  (len(Bridge)-1) x n, containing  n MC samples for the contribution from x to y (columns of files)
#K is a list of size (len(Bridge)-1) x n, containing n MC samples for the K sampled points (columns of files)
#ome is a list of size (len(Bridge)-1) x n containing n MC arrays of size K with sampled WFB candidates
def min_logLikelihood (h,*params):
    N,Bridge, T_star, theta, K, ome, A , phi, phi_max, phi_minus= params#pMl, 
    MC_conts=np.ones(len(Bridge)-1)#where len(Bridge)-1 is the number of contributions
    conts=np.zeros(len(Bridge)-1)
    for i in range(len(Bridge)-1):
        MC_conts[i]= L_cont_a (N, K[i],ome[i],theta,h, phi, phi_max, phi_minus)
        if (MC_conts[i]==0.):
            conts[i]=1
        else:
            conts[i]= np.exp( A( Bridge[i+1], Bridge[i] , h) - \
                 (T_star[i+1]-T_star[i])*phi_minus(h, theta) ) * MC_conts[i]
        #print("MC_conts=", MC_conts, "\n")
        if (conts[i]<=0):
            warnings.warn("Log of negative not possible!")
            #print("MC_conts=", MC_conts, "\n")
        conts[i]= np.log( conts[i] ) #np.log(np.mean(pMl[i]))+ 
    #print("conts=", conts, "\n")
    Loglikelihood=(-1)*(sum(conts ) )
    return Loglikelihood


