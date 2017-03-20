import scipy.stats as ss 
import numpy as np
import numpy.random as npr 
from math import sqrt

def sample(G, D, A, B, n):
    P = {}
    P['mu']     = npr.normal(loc=G,scale=1./sqrt(D))
    P['lam']    = npr.gamma(shape=A,scale=1./B)
    X           = npr.normal(loc=P['mu'],scale=1./sqrt(P['lam']), size=n)
    return X, P

class simpleHierarchicalModel:
    '''A simplified hierarchical model.'''
    def __init__(self, observedData, G, D, A, B):
        self.G = G
        self.D = D
        self.A = A
        self.B = B
        self.X    = observedData
        self.n    = len(self.X)
        self.mean = self.X.mean()
        self.var  = self.X.var()
        self.params = ['mu', 'lam']
        self.rv     = {'mu': self.rv_mu, 'lam':self.rv_lam }
        
    def rv_lam(self, lam, mu):
        denominator = (self.B+0.5*self.n*(self.var+(mu-self.mean)**2))
        return npr.gamma(shape=self.n/2. + self.A, scale=1./denominator)
    
    def rv_mu(self, lam, mu):
        denominator = self.n*lam + self.D
        return npr.normal(loc=(self.n*lam*self.mean+self.D*self.G)/denominator,scale=1./denominator)