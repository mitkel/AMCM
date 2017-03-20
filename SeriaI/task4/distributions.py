import numpy as np
import numpy.random as npr 
from math import sqrt, log, fsum, gamma, pi


class distribution:
    def __init__(self):
        raise NotImplementedError

    def unnorm_log_pdf(self, *params):
        raise NotImplementedError

    def rvState(self, *params):
        pass

class multivariate_t_student(distribution):
    def __init__(self, df, N):
        self.df = df
        self.dim = N

    def rvState(self):
        return npr.standard_t(self.df, size=self.dim)

    def unnorm_log_pdf(self, *State):
        logQ = 0
        logQ = -fsum((self.df + 1)/2*log(1+a**2/float(self.df)) for a in State)
        #+ stała normująca
        logQ = logQ + self.dim * (log(gamma((self.df+1)/2)) - log(sqrt(pi * self.df)) - log(gamma(self.df/2)))
        return logQ

class multivariate_normal(distribution):
    def __init__(self, mean_mtr, dev_mtr, N):
        self.mean_mtr = mean_mtr
        self.dev_mtr = dev_mtr
        self.dim = N

    def unnorm_log_pdf(self, *State):
        inverse_dev = np.linalg.inv(self.dev_mtr)
        logQ = 0
        logQ -= fsum(0.5*inverse_dev[a,b]*(State[a]-self.mean_mtr[a])*(State[b]-self.mean_mtr[b]) \
            for a, b in zip(range(self.dim), range(self.dim)))  
        logQ = logQ - self.dim/2*log(2*pi) -2*log(np.linalg.det(self.dev_mtr)) #dodanie stałej normującej
        return logQ
