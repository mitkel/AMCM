import scipy.stats as ss 
import numpy as np
import numpy.random as npr 
from math import sqrt, log


class hierarchy:
    def __init__(self, G, D, A, B):
        self.G = G
        self.D = D
        self.A = A
        self.B = B

    def unnorm_log_pdf(self, mu, lam):
        # implement the density of pair (mu, lam) given the hyperparameters (the middle level of the hierarchy in the model)
        return (log(lam)*(self.A-1)-lam*self.B) - self.D*((mu - self.G)**2)/2.

class hierarchy_conditional(hierarchy):
    '''A simplified hierarchical model.'''
    def __init__(self, observedData, G, D, A, B):
        super(hierarchy_conditional,self).__init__(G, D, A, B)
        self.X      = observedData
        self.n      = len(self.X)
        self.mean   = self.X.mean()
        self.var    = self.X.var()
        self.params = ['mu', 'lam']
        self.rv = dict( (x, getattr(self, 'rv_'+x) ) for x in self.params ) # This approach uses my naming convention for random variable generators  

    def rv_lam(self, lam, mu):
        denominator = (self.B+0.5*self.n*lam*(self.var+(mu-self.mean)**2))
        return npr.gamma(shape=self.n/2. + self.A, scale=1./denominator)
        
    def rv_mu(self, lam, mu):
        denominator = self.n*lam + self.D
        return npr.normal(loc=(self.n*lam*self.mean+self.D*self.G)/denominator,scale=1./sqrt(denominator))
        
    def unnorm_log_pdf(self, lam, mu):
        res = super(hierarchy_conditional, self).unnorm_log_pdf(lam, mu) 
        res += log(lam)*self.n/2. - self.n/2.*lam*(self.var + (mu - self.mean)**2)
        return res

class hierarchy_full(hierarchy):
    def __init__(self, G, D, A, B, n):
        super(hierarchy_full, self).__init__(G, D, A, B)
        self.n = n

    def rvState(self):
        '''Returns a random state given the hyperparameters.'''
        P = {}
        P['mu']     = npr.normal(loc=self.G,scale=1./sqrt(self.D))
        P['lam']    = npr.gamma(shape=self.A,scale=1./self.B)
        return P

    def rvState_and_observations(self):
        '''Returns both a random state and data given the hyperparameters.'''
        P = {}
        P['mu']     = npr.normal(loc=self.G,scale=1./sqrt(self.D))
        P['lam']    = npr.gamma(shape=self.A,scale=1./self.B)
        X           = npr.normal(loc=P['mu'],scale=1./sqrt(P['lam']), size=self.n)
        return P, X
