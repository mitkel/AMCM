import scipy.stats as ss 
import numpy as np
import numpy.random as npr 
from math import sqrt
from test import merge_dicts

def generateSample(G, D, A, B, n):
    P = {}
    P['mu']     = npr.normal(loc=G,scale=1./sqrt(D))
    P['lam']    = npr.gamma(shape=A,scale=1./B)      # model parameters are mu and lam
    X           = npr.normal(loc=P['mu'],scale=1./sqrt(P['lam']), size=n)      # The sample we will observe.    
    return X, P

def rLam(var, mean, n, lam, mu, G, D, A, B): 
    return npr.gamma(shape=n/2. + A,scale=1./(B+0.5*n*(var+(mu-mean)**2)))
    
def rMu(var, mean, n, lam, mu, G, D, A, B): #mu ~ N(nlam/(nlam+D)mean + D/(nlam+D)G, 1/(nlam + D))
    return npr.normal(loc =(n*lam*mean+D*G)/(n*lam+D),scale=1./(n*lam + D))

marginals = { 'lam':rLam, 'mu':rMu }

def systematicUpdate(Y, params, hyper):
    for m in marginals:
        params[m] = marginals[m](**merge_dicts(Y, params, hyper)) 
    return params

def randomUpdate(Y, params, hyper):
    for m in npr.choice(list(params.keys()),1):
        params[m] = marginals[m](**merge_dicts(Y, params, hyper)) 
    return params

def Gibbs(Y, params, hyper, update, simLen = 5000):
    states = [ params.copy() ]
    for k in range(simLen):
        params = update(Y,params,hyper)
        states.append(params.copy())
    return states
