from MCalgo import MCalgo
import scipy.stats as ss
import numpy as np
from math import log
from itertools import chain

class MetropolisHastings(MCalgo):
    def __init__(self, stepsNo, burnIn, target, state0, proposal):
        super(MetropolisHastings, self).__init__(stepsNo, burnIn, target, state0)
        self.proposal = proposal

    def updateState(self):
        prop = self.proposal.rv( **self.state )
        logalpha = self.logAlpha(prop)
        if logalpha == -np.inf or np.isnan(logalpha):
            return
        elif logalpha == np.inf:
            self.state = prop.copy()
        else:
            logU = log( ss.uniform.rvs() )
            if logalpha > logU:
                self.state = prop.copy()

    def logAlpha(self, prop):
        logalpha = \
            self.target.logUnnormPDF(**prop) - \
            self.target.logUnnormPDF(**self.state) + \
            self.proposal.q(*tuple(chain([prop[k] for k in self.keys],[self.state[k] for k in self.keys]))) - \
            self.proposal.q(*tuple(chain([self.state[k] for k in self.keys],[prop[k] for k in self.keys])))
        return logalpha

    def Name(self):
        return "Metropolis Hastings"