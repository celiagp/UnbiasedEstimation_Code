#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:41:37 2022

@author: celitona
"""
#%%
import numpy as np
import Auxiliary_NeutralWF as ANWF #For neutral WF candidates


# =============================================================================
#Exact algorithm for Coupled WF diff with L loci and 2 allele type
##Parameters:
##All numerical parameters need to be of np.array type
###x Vector of initial values (not random)
###y Vector of initial values (not random)
###t time in-between x and y
###phi.plus Upper bound for phi
###phi.minus Lower bound for phi
###phi.FUNC function, with phi.minus yet to be extracted, 
#####with arguments: xs matrix of points, theta matrix of mutation parameters and h vector of interac parameters 
###A.FUNC function with arguments: xs vector of XT points, theta vector of mut parameters and h vector of interac parameters 
###A.plus Upper bound for A.FUNC
###theta matrix of neutral drift parameters
###h vector of interaction parameters
#Value: List
###ts Sampled ts (also with 0 and T)
###xs Matrix of sampled accepted skeletons for solution 1
###count Number of attempts before a skeleton is accepted
###N Number of Poisson points in the accepted path
###coefs Number of coefficients needed in b for simulating the candidate (L dimensional with number of candidates)
# =============================================================================

#%%

def Exact_WFb_est(X,Y,t,phi_max,phi_minus,phi_FUNC,theta, h):
    accept, count, K_tot = 0, 0, 0#(Rejection/Acceptance flag), Number of attempts, Total number of sampled Poisson points
    #coefs = np.zeros(len(x))#Number of coefficients needed in b for simulating the candidates: dimension L number of Loci
    while accept != 1: #1)Simulate PoissonProcess in [t0,T]x[0,phi.plus-phi.minus]
        lmbda = phi_max (h, theta)
        t0=0
        area = (t-t0)*(1-0)
        K = np.random.poisson(lmbda*area)
        x = np.random.uniform(t0, t, K)
        y = np.random.uniform(0,1, K)
        xpoints = x[np.argsort(x)]
        ypoints = y[np.argsort(x)]
        #print("K=", K, "x=", X,"y=",Y, "xpoints=",xpoints, "ypoints", ypoints)
        if K!=0: #3.1) Given a Poisson process realization "with points" #Initialize at t1
            ts = np.concatenate( ([t0], xpoints, [t]) )#t points to pass (as differences) to NeutralWF
            xs = np.concatenate( ([X], np.zeros(len(xpoints)), [Y]) )#vector with row vectors of xsi points to pass to NeutralWF starting with x0i
            i = 0#Index initialization
            while i<(len(ts)-1):#length ts is N+2= ncol of xs matrix
                AUX = ANWF.Neutral_WF( theta, ANWF.b, ANWF.a, ANWF.S_minus,
                ANWF.S_plus, xs[i], ts[i+1]-ts[i], ANWF.griffiths_approx)
                xs[ i+1] = AUX[0]#Candidate sample of WF diffusion k at ts[i+1]
                i = i+1 #Now column vectors in xs contain the Neutral WF candidates for corresponding to t0, t1, t2, ...., T (note that x0 is not really a candidate)
            
            
            C = xs[1:-1]#copy of matrix xs from the second to the next-to-last column: Poisson points
            #"A rej=", np.exp(A_FUNC(xs[:,1], x0, h)-A_plus)) 
            #print("ts=", ts, "xs=", xs, "C", C)
            flag = [1 if np.all( ( (phi_FUNC(C,theta,h) - phi_minus(h, theta))/ phi_max (h, theta)) <=ypoints) else 0] #4.1) Rejection scheme
            #print("flag=", flag)
            if flag == [1]: print("accepted!")
        else:#3.2)Given a Poisson process realization "with NO points"
            ts = np.append(t0,t)
            xs = np.append( X,Y)
            flag = [1]
            #print("flag=", flag)
        if flag == [1]:#Response to flag is array
                accept = 1
        else:
            accept = 0
        count += 1
        K_tot += K
    return ([ts, xs, count, K_tot, K])

#%%

def Exact_WFb_gen(X,t,phi_plus,phi_minus,phi_FUNC,A_FUNC, A_plus,theta, h):
    accept, count, K_tot = 0, 0, 0#(Rejection/Acceptance flag), Number of attempts, Total number of sampled Poisson points
    while accept != 1: #1)Simulate PoissonProcess in [t0,T]x[0,phi.plus-phi.minus]
        lmbda = phi_plus (h, theta)-phi_minus(h, theta)
        t0=0
        area = (t-t0)*(1-0)
        K = np.random.poisson(lmbda*area)
        x = np.random.uniform(t0, t, K)
        y = np.random.uniform(0,1, K)
        xpoints = x[np.argsort(x)]
        ypoints = y[np.argsort(x)]
        u = np.random.uniform(0, 1)#2) Simulate Uniform 
        #print("K=", K, "x=", X,"y=",Y, "xpoints=",xpoints, "ypoints", ypoints)
        if K!=0: #3.1) Given a Poisson process realization "with points" #Initialize at t1
            ts = np.concatenate( ([t0], xpoints, [t]) )#t points to pass (as differences) to NeutralWF
            xs = np.concatenate( ([X], np.zeros(len(xpoints)+1)) )#vector with row vectors of xsi points to pass to NeutralWF starting with x0i
            i = 0#Index initialization
            while i<(len(ts)-1):#length ts is N+2= ncol of xs matrix
                AUX = ANWF.Neutral_WF( theta, ANWF.b, ANWF.a, ANWF.S_minus,
                ANWF.S_plus, xs[i], ts[i+1]-ts[i], ANWF.griffiths_approx)
                xs[ i+1] = AUX[0]#Candidate sample of WF diffusion k at ts[i+1]
                i = i+1 #Now column vectors in xs contain the Neutral WF candidates for corresponding to t0, t1, t2, ...., T (note that x0 is not really a candidate)
            C = xs[1:-1]#copy of matrix xs from the second to the next-to-last column: Poisson points
            #print("ts=", ts, "xs=", xs, "C", C)
            flag = [1 if np.all( ( (phi_FUNC(C,theta,h) - phi_minus(h, theta))\
                /(phi_plus (h, theta) - phi_minus(h,theta))) <=ypoints) and np.all((np.exp(A_FUNC(xs[-1], X, h)-\
            -A_plus(h)))>=u)else 0] #4.1) Rejection scheme
            #print("flag=", flag)
        else:#3.2)Given a Poisson process realization "with NO points"
            ts = np.append(t0,t)
            xs = np.append( X, 0 )
            AUX = ANWF.Neutral_WF( theta, ANWF.b, ANWF.a, ANWF.S_minus,
            ANWF.S_plus, xs[0], ts[1]-ts[0], ANWF.griffiths_approx)
            xs[1] = AUX[0]
            flag = [1 if np.all((np.exp(A_FUNC(xs[-1], X, h)-A_plus(h)))>=u)else 0]
            #print("flag=", flag)
        if flag == [1]:#Response to flag is array
                accept = 1
        else:
            accept = 0
        count += 1
        K_tot += K
    return ([ts, xs, count, K_tot, K])