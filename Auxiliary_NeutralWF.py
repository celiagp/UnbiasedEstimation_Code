#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:58:49 2022

@author: celitona
"""
#%%
import numpy as np
import warnings
from scipy.special import factorial
from scipy.stats import beta
from scipy.stats import binom
from scipy.stats import norm
# =============================================================================
#Auxiliary Functions for Neutral Wright Fisher diffusion simulation####
####Neutral diffusion model for the 2 alleles case: dXt=alpha(Xt)*dt+sqrt(Xt*(1-Xt))dBt , 
#with alpha(Xt)=0.5*[theta1*(1-Xt)-theta2*Xt]=0.5[theta1-(theta1+theta2)Xt] and theta=theta1+theta2
#with transition density function
#f(x,y;t)=\sum_(m=0)^infty q_m^theta (t) \sum_(l=0)^m binom(m,x) (l) beta(theta1+l,theta2+m-l) (y)
####Transition function for the death process A_infty^theta (t) is: 
#q_m^theta (t) = sum_(k=m)^infty (-1)^(k-m) b_k^(t,theta) (m), with
#b_k^(t,theta) (m) = a_(km)^theta * exp(-k(k+theta-1)t/2), with
#a_(km)^theta=G1[Gamma(G2+y)/Gamma(G2)]/m!(k-m)!, with
#G1=(theta+2k-1)
#G2=(theta+m) and y=k-1
# =============================================================================
####a function
#Parameters:
###theta sum of neutral mutation drift parameters 
###m realization of the random variable to be sampled
###k counting index
#Value: 
###a function evaluated at parameters
###recall that a=(theta+2*k-1)gamma(theta+m+k-1)/gamma(theta+m)m!(k-m!)
# =============================================================================
def a( theta, m, k):
    #Flags:
    #1.Simplifying computation if G1==0
    G1 = theta+2*k-1
    if G1==0:
        return(0)
    elif k<m or m<0:
        warnings.warn("k can not be smaller than m")
        return(0)
    G2 = theta+m
    denom = factorial(m)*factorial(k-m)#We are not accounting for possible numerical errors. Seems that Python handles it well
    #Note now that that gamma(G2+k-1)/gamma(G2)=G2*(G2+1)*....*(G2+k-2), so,
    #num<-G1*gamma(G2+k-1)/gamma(G2)=G1*G2*(G2+1)*....*(G2+k-2), with exceptions for k=0 and k=1
    if np.isinf(denom):
        denom = 1e+308
    if k == 0:#Note that when k==0 then m==0
        num = 1
    elif k == 1:#This works for k==1 and both m==0 and m==1
        num = theta + 1
    elif k >= 2:
        PROD = 1
        for i in range(2, k+1):
            PROD *= (G2+i-1)
        num = G1*PROD
        #print(" PROD= ", PROD)
    if np.isinf(num):
        num = 1e+308
    #print ("G1= ", G1, " G2=", G2, " num=", num, " denom= ", denom)
    return num/denom #Watch out that num and denom can be Inf or even Nan. In both cases it is because they are too big. Controlled by in b?
# =============================================================================
#b function
#Parameters: (parameter m enters through function a, i.e., for each m we have a different value of a)
###a_FUNC vectorised function a (to evaluate at parameters)
###theta sum of neutral drift parameters 
###m realization of the random variable to be sampled
###k counting index
###t time instant
#Value: 
###b function evaluated at parameters
# =============================================================================
def b( a_FUNC, theta, m, k, t):#Watch out that a can be infinity or NAN. Look at how to check in Python 
    a = a_FUNC( theta, m, k)
    if np.isinf(a):
        a = 1e+308
        warnings.warn("Variable a was too big")
    return a*np.exp(-k*(k+theta-1)*t/2)#Not checked if a is OK

# =============================================================================
#S functions
#Parameters:
###b.FUNC function b to be evaluated at parameters
###a.FUNC function a (to evaluate at parameters)
###theta sum of  neutral drift parameters 
###t time instant
###k vector of coefficients to be computed for each M (depends on M and has (increasing) varying length)
###M realization of the random variable to be sampled (upper limit of the outer sum). Check Jenkins and Spanò paper
#Value: 
###S.minus/S.plus function evaluated at parameters
# =============================================================================

def S_minus( b_FUNC, a_FUNC, theta, M, k, t):
    SUM = 0
    for j in range(M+1):#Dimension of vector k is always M+1 because M starts at 0 and vectors in R start at [1]
        for i in range(2*k[j]+2):
            SUM += (-1)**i*b( a_FUNC, theta, j, j+i, t)
    return SUM

def S_plus( b_FUNC, a_FUNC, theta, M, k, t):
    SUM = 0
    for j in range(M+1):#Dimension of vector k is always M+1 because M starts at 0 and vectors in R start at [1]
        for i in range(2*k[j]+1):
            SUM += (-1)**i*b( a_FUNC, theta, j, j+i, t)
    return SUM
# =============================================================================
#Sampling from the mixture 
#f(x,y;t)=\sum_(m=0)^infty q_m^theta (t) \sum_(l=0)^m binom(m,x) (l) beta(theta1+l,theta2+m-l) (y)

#Parameters:
###theta1 and theta2, neutral drift parameters 
###griffiths.approx sampling approximation for q_m^theta (t) (asymptotically normal, check Jenkins and Spano pg 15)
###t time instant in which we want to sample the realization of the path
#Value: List
###m sampled realization of the Neural WF diffusion at t
# =============================================================================

def griffiths_approx( theta, t):
    Beta = 0.5*(theta-1)*t
    if Beta==0:
        mu = 2/t
        sigma = 2/(3*t)
    else:
        eta = Beta/( np.exp(Beta)-1 )
        mu = 2*eta/t
        sigma = 2*eta/t * (eta+Beta)**2 * (1+ eta/( eta+Beta ) - 2*eta) * Beta**(-2)
    m = norm.rvs(mu,sigma**(1/2))
    #print("m", m,"\n")
    m = int(np.round(np.absolute(m)))
    return m 
# =============================================================================
#Sampling from the transition function q_m^theta (t) of the death process A_infty^theta (t)
#Parameters:
###theta sum of neutral drift parameters 
###b.FUNC function b to be evaluated at parameters
###a.FUNC function a (to evaluate at parameters)
###S.minus function (to evaluate at parameters)
###S.plus function (to evaluate at parameters)
###t time instant
#Value: List
###m sampled realization of the death process
###sum(k) sumber of computed coefficients
# =============================================================================
def death_process ( theta, b_FUNC, a_FUNC, S_minus, S_plus, t, griffiths_approx):
    #M<-M_init<-0 #Instead of starting at M=0, we start at the approximation of mode(q_m^theta) found in Th 1 Jenkins and Spanò
    #k<-0
    if t< 0.05:#Griffiths approximation
        M = griffiths_approx( theta, t)
        warnings.warn("Griffiths approx use because t smaller than 0.05")
        return np.append(M, 0)
    Beta = 0.5*(theta-1)*t
    if Beta == 0:#Make sure Beta==0 works always
        eta = 1
    else:
        eta = Beta/(np.exp(Beta)-1)
    M, M_init = max( 0, round(2*eta/t) ), max( 0, round(2*eta/t) )#We are using 
    #an approximation for the mode of q_m^theta(t), which is the mean of Griffiths Normal 
    #approximation, rounded to the nearest positive integer
    k, k1 = np.zeros( M_init+1, dtype = int), np.zeros( M_init+1, dtype = int)
    it, teles =  0, 1#k always has length M+1,Iteration number index, Telescopic index for 
    #pivoting from M_init  
    u =  np.random.uniform(0, 1)
    do_teles = True
    while True:#Note that this condition is always met
        #Find infimum term in which the decay of the series begins (depending on M). 
        #Instead of comparing bj+M+1 with bj+M, we compare
        #while(b.FUNC( a.FUNC, theta, M, j+M+1, t)>=b.FUNC( a.FUNC, theta, M, j+M, t))
        #{bj+M+1/bj+M=A*B with 1. It has to be done for ALL ks that are initalized at 0!
        for i in range(len(k)):
            j, m  = 0, i 
            #j is the summation index for function b (terms to be considered)
            #i index = value of m
            if j+m == 0:#Special case of A with k=0
                A = theta + 1
            elif j+m == 1:#Special case of A with k=1
                A = ( (theta+3)*(theta+m) ) / \
                    ( factorial(m)*factorial(2-m)*(theta+1) )
            else:#Computing A if (j+m) !=0 and (j+m) !=1
                A = ( (theta+2*(j+m)+1)*(theta+m+(j+m)-1) )/ \
                    ( (theta+2*(j+m)-1)*((j+m)+1-m) )
            B = np.exp( -(2*(j+m)+theta)*t/2 )#Computing B
            while A*B>=1:#Comparison to find first term in which decay starts
                j +=1
                if j+m == 0:#Special case of A with k=0
                    A = theta+1
                elif j+m == 1:#Special case of A with k=1
                    A = ( (theta+3)*(theta+m) ) / \
                        ( factorial(m)*factorial(2-m)*(theta+1) )
                else:#Computing A if (j+m) !=0 and (j+m) !=1
                    A = ( (theta+2*(j+m)+1)*(theta+m+(j+m)-1) )/ \
                        ( (theta+2*(j+m)-1)*((j+m)+1-m) )
                B = np.exp( -(2*(j+m)+theta)*t/2 )#Computing B
            if k[i]<np.ceil( j/2 ):#Update only the ks that have not yet hit the decaying zone
                k[i] = np.ceil( j/2 )
        #Initialize S functions values. If M or k are too high, use Griffiths
        if M> 50 or np.any(k>50):#Griffiths approximation
            M = griffiths_approx( theta, t)
            warnings.warn("Griffiths approx use because M or k larger than 50")
            return np.append(M, 0)
        sm = S_minus( b_FUNC, a_FUNC, theta, M, k, t)
        SM = S_plus( b_FUNC, a_FUNC, theta, M, k, t)
        #print ("M_init= ", M_init, " M beg=", M, "k=", k, "k1= ", k1, " j=", j, \
               #" i=", i , " it=", it, " u=", u, " sm=", sm, " SM=", SM )
        while sm < u and u < SM:
            k+=np.ones(len(k), dtype = int)#Update of number of coefficients
            sm = S_minus( b_FUNC, a_FUNC, theta, M, k, t)#update S.minus
            SM = S_plus( b_FUNC, a_FUNC, theta, M, k, t)#update S.plus
        if sm>u:#We are done!
            #print ("M_init= ", M_init, "Returning M=", M, "k=", k, "k1= ", k1, " j=", j, \
                   #" i=", i , " it=", it, " u=", u, " sm=", sm, " SM=", SM )
            return np.append(M, sum(k)+len(k))
            # return np.array([M, sum(k)+len(k)], dtype = int)
        #else:
        if SM < u:#We update M and the vector k
            if ( M_init+(-1)**(it)*teles )<0 :
                do_teles = False
                M += 1
                print("M", M, "M_init", M_init, "k=", k, "k1=", k1, "it=", it, "\n")
                k = np.append( k[: M], 0, dtype = int )
            if do_teles:#We radiate outwards from an initial approximation of M_init,
                #unless we have hit already M=0. dim of k always stays M+1, with 0 in 
                #the last(s) position(s) when we start over
                M = M_init+(-1)**(it)*teles
                    #M will be then: c(M_init,M_init+1, M_init-1, M_init+2, M_init-2,M_init+3,M_init-3)
                #print ("M_init= ", M_init, " M=", M, "k=", k, "k1= ", k1, " j=", j, \
                       #" i=", i , " it=", it, " u=", u, " sm=", sm, " SM=", SM )
                for l in range( min(len(k), len(k1) ) ):
                        #We save k in an auxiliary vector before we do any modifications.
                    if k1[l]<k[l]:
                        k1[l] = k[l]
                    if len(k)>len(k1):
                        k1 = np.append( k1, k[ len(k1): ] )# We take the elements of k beyond the length of k1. 
                            #From R: c(k1,tail(k,length(k)-length(k1)))
                if it%2 == 0: #For even it we lengthen k and add a 0s at the end, M has already been upodated!
                    k = np.append( k1, np.zeros( (M+1)-len(k1), dtype = int))
                else:#For odd it we shorten k, M has already been updated!
                    k = k1[:M+1] #First M+1 (from 0 to M) terms
                    #print("k=", k, "k1=", k1, "it=", it)
        if it%2 == 1:
            teles +=1
        it +=1
# =============================================================================
#Neutral Wright Fisher diffusion Jenkins and Spanò with 2 allele types####
####Neutral diffusion model: dXt=alpha(Xt)dt+sqrt(Xt(1-Xt))dBt , 
#with alpha(Xt)=0.5*[theta1*(1-Xt)-theta2*Xt]= 0.5*[theta1-theta*Xt]and theta=theta1+theta2
#with transition density function 
#f(x,y;t)=\sum_(m=0)^infty q_m^theta (t) \sum_(l=0)^m binom(m,x) (l) beta(theta1+l,theta2+m-l) (y)

#Sampling from the mixture 
#f(x,y;t)=\sum_(m=0)^infty q_m^theta (t) \sum_(l=0)^m binom(m,x) (l) beta(theta1+l,theta2+m-l) (y)
#Given x, we sample y occurring after time t(transition probability density)
#Parameters:
###theta vector of drift's neutral mutation parameters 
###b.FUNC function b to be evaluated at parameters
###a.FUNC function a (to evaluate at parameters)
###S.minus function (to evaluate at parameters)
###S.plus function (to evaluate at parameters)
###x previous ("initial") realization of the WF at "another" instant, where we want the path to start
###griffiths.approx function for using griffiths approximation when t<0.05
###t time increment after x in which we want to sample the realization of the path y_t
#Value: List
###y sampled realization of the Neural WF diffusion at "length t" after the time realization for x
###m[1]=sum(k) (second returned argument from death process) which counts the number of coefficients needed to return m[0]
# =============================================================================
def Neutral_WF( theta, b_FUNC, a_FUNC, S_minus, S_plus, x, t, griffiths_approx):
    #m = death_process( np.sum(theta), b_FUNC, a_FUNC, S_minus, S_plus, t, griffiths_approx) 
    m=griffiths_approx( np.sum(theta), t)
    m=np.array([m])
    l = binom.rvs( m[0], x)#np.random.binomial( m[0], x)
    #print("a= ", theta1+l, "b=",theta2+m[0]-l,  "Sample Size= ", np.broadcast (theta1+l,theta2+m-l ).size )
    y = beta.rvs(theta[0]+l,theta[1]+m[0]-l)#np.random.beta( (theta[0]+l), (theta[1]+m[0]-l))
    #print("m=", m[0], "coefs", m[1]," x=", x, " l=", l, " y= ", y, "AUX= ", np.append(y,m[1]))
    #return np.append(y,m[1])
    return np.append(y,0)
# =============================================================================