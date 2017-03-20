import numpy.random as npr
import scipy.stats as ss
from math import log

class Q():
	def __init__(self, alpha):
		self.alpha = alpha

	def q( self, stateNew, stateOld): # jadro przejscia jest symetryczne
		PropositionPdf 	= 1
		return PropositionPdf

	def rv(self, state):
		thetaNew = npr.uniform(-self.alpha, self.alpha) + state
		return thetaNew

# zakladam prior z produktowego rozkladu jednostajnego U(0,20)
class pseudo_target():
	def __init__(self, pseudo, target):
		self.pseudo = pseudo
		self.target = target
	def UnnormPDF(self, state):
		return self.pseudo(state) + self.target(state)


def Exp1(theta):
	return npr.exponential(scale = 1)
def Exp2(theta):
	return npr.exponential(scale = 2)
def Norm11(theta):
	return npr.normal(loc = 1, scale = 1)
def Norm01(theta):
	return npr.normal(loc = 0, scale = 1)
def Norm0t(theta):
	return npr.normal(loc = 0, scale = .1 + 10*theta**2)
def target(theta):
	return ss.norm.pdf(theta, loc = 0, scale = 1)