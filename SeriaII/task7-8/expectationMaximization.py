from target import target
from math import exp
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

w   = .1
m1  = 0
m2  = 5
N   = 1000
mc_steps = 100000
state0 = (w, m1, m2)

T       = target(mc_steps, N)
sample  = list( T.rvs(N, *state0) )

sns.distplot(sample, hist=True, rug=True)
plt.show()


class ExpectationMaximization:
    def __init__(self, sample):
        self.X = sample
        self.N = len(sample)

    def p_one(self, x, w, m1, m2):
        A = w * exp(-(x-m1)**2/2)
        B = (1-w) *  exp(-(x-m2)**2/2)
        return A/(A+B)

    def p(self, w, m1, m2):
        P = [ self.p_one(x, w, m1, m2) for x in self.X ]
        return P

    def new_theta(self, w, m1, m2):
        P       = self.p( w, m1, m2)
        sumP    = sum(P)
        new_w   = sumP/self.N
        cross   = sum( x*p for x,p in zip(self.X, P) )
        new_m1  = cross/sumP
        new_m2  = (sum(self.X)-cross)/(self.N - sumP)
        return new_w, new_m1, new_m2

    def run(self, w, m1, m2, iter):
        for _ in range(iter):
            w, m1, m2 = self.new_theta(w, m1, m2)

        return w, m1, m2

EM = ExpectationMaximization(sample)
EM.run( .5, 4, 5, 1000 )
