# Here you should implement the delayed acceptance.
# It can be a daughter class of MCalgo class.

from MCalgo import MCalgo
import scipy.stats as ss
import numpy as np
from numpy import log
from itertools import chain

class DelayedAcceptance(MCalgo):
	def __init__(self, stepsNo, burnIn, target, state0, proposal, levels):
		super(DelayedAcceptance, self).__init__(stepsNo, burnIn, target, state0)
		self.proposal 	= proposal
		self.levels		= levels

	def Name(self):
		return "Delayed Acceptance MH"

	def updateState(self):
		prop = self.proposal.rv( **self.state )
		logU = log( ss.uniform.rvs(size = self.levels))
		accepted = True
		for level in range(self.levels):
			logalpha = self.logAlpha( prop, level )
			if logalpha == -np.inf or np.isnan(logalpha) or (not np.isinf(logalpha) and logalpha < logU[level]):
				accepted = False
		if accepted:
			self.state = prop.copy()

	def logAlpha(self, prop, level):
		logalpha = \
			getattr(self.target, 'logUnnormPDF' + str(level+1))(**prop) - \
			getattr(self.target, 'logUnnormPDF' + str(level+1))(**self.state)
		if level+1 == self.levels:
			logalpha = logalpha + \
				self.proposal.q(*tuple(chain([prop[k] for k in self.keys],[self.state[k] for k in self.keys]))) - \
				self.proposal.q(*tuple(chain([self.state[k] for k in self.keys],[prop[k] for k in self.keys])))
		return logalpha

