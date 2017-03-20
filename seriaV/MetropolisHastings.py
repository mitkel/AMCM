from MCalgo import MCalgo
import scipy.stats as ss
from math import log, exp

class MetropolisHastings(MCalgo):
    def __init__(self, stepsNo, burnIn, target, state0, proposal):
        super(MetropolisHastings, self).__init__(stepsNo, burnIn, target, state0)
        self.proposal = proposal

    def updateState(self):
        prop = self.proposal.rv( self.state )
        U = ss.uniform.rvs()
        if self.Alpha(prop) > U:
            self.state = prop

    def Alpha(self, prop):
        nom = self.target.UnnormPDF(prop)
        den = self.target.UnnormPDF(self.state)
        if den == 0:
            return 1
        elif nom == 0:
            return 0
        else:
            return exp(log(nom)-log(den))