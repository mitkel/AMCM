from MCalgo import MCalgo
import scipy.stats as ss
import numpy.random as npr
from math import log

class MH_within_Gibbs(MCalgo):
    def __init__(self, stepsNo, burnIn, state0, target, proposal, gibbs, mParams, gParams):
        super(MH_within_Gibbs, self).__init__(stepsNo, burnIn, target, state0)
        self.proposal   = proposal  # jądro przejścia w MH
        self.gibbs      = gibbs     # funkcje losujące zmienne z kroku gibbsa
        self.mParams = mParams # dictionary with MH params
        self.gParams = gParams # dictionary with gibbs params 

    def updateState(self):
        self.metropolisUpdate()
        self.gibbsUpdate()

    def metropolisUpdate(self):
        prop = self.state.copy()
        prop.update(self.proposal.rv( self.state ))
        logU = log( ss.uniform.rvs() )
        if self.logAlpha(prop) > logU:
            self.state.update(prop)

    def gibbsUpdate(self):
        raise NotImplementedError

    def logAlpha(self, prop):
        logalpha = \
            self.target.logUnnormPDF(prop) - \
            self.target.logUnnormPDF(self.state) + \
            self.proposal.q(self.state, prop) - \
            self.proposal.q(prop, self.state)
        return logalpha

class MH_within_Gibbs_RS(MH_within_Gibbs):
    def gibbsUpdate(self):
        for m in npr.choice(self.gParams, size = 1):
            self.state[m] = self.gibbs.rv[m](self.state)

class MH_within_Gibbs_DS(MH_within_Gibbs):
    def gibbsUpdate(self):
        for m in self.gParams:
            self.state[m] = self.gibbs.rv[m](self.state)
