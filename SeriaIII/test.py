import numpy as np
import numpy.random as npr
import scipy.stats as ss
from math import exp, log
X= [1,2,3,4,5,4.8,9,1,1,4,5,6,3,2.2,3,4,1]
m = [1,2,3,4,5]
w = [1,0,0,0,0]
Y= []
K = 5
Y = np.zeros(shape = (K, len(X)) ) #losujemy początkowe Y
for i in range(len(X)):
	Y[npr.choice(K)][i] = 1
# print(np.shape(Y))
# print(Y)


stateOld = {'X':X, 'Y':Y, 'w': w, 'm':m}

def lognormTarget(state):
    s = 0
    for i in range(np.shape(state['Y'])[0]):
        for j in range(np.shape(state['Y'])[1]):
            s = s-0.5*((state['X'][j]-state['m'][i])**2)*state['Y'][i][j]
    return s

# print(Y)
# print(lognormTarget(stateOld))


def rvW(state):
	alpha = []
	K = len(state['w'])
	[(alpha.append(1+sum(state['Y'][i]))) for i in range(K)]
	return {'w':npr.dirichlet(alpha)}
# w = rmW(stateOld)['w']
# print(Y)
# print(w)

def rvY(state):
	K = len(state['w'])
	N = len(state['X'])
	Ynew = np.zeros(shape = (K, N) )
	for i in range(N):
		p = np.zeros(K)
		s = 0
		for j in range(K):
			p[j] = state['w'][j]*exp(-0.5*(state['X'][i]-state['m'][j])**2)
			s = s + p[j]
		p = p/s # normalizacja
		Ynew[npr.choice(K,p=p)][i] = 1
	return {'Y': Ynew}

# print(rvY(stateOld))



class Q():
	def __init__(self, K):
		self.K = K

	def q( self, stateOld, stateNew):
		logPropositionPdf 	= 0
		logStartingPdf 		= 0
		for i in range(self.K):
			if sum(stateOld['Y'][i]) == 0:
				p = 0.0001
			else:
				nominator = sum([(x*y) for x,y in zip(stateOld['X'],stateOld['Y'][i])])
				denominator = 20*sum(stateOld['Y'][i])
				p = nominator/denominator
			alpha = p/min(p,1.-p)
			beta  = (1.-p)/min(p,1.-p)

			logPropositionPdf	=	logPropositionPdf 	+ 	log(ss.beta.pdf( stateNew['m'][i]/20, a = alpha, b = beta))
			logStartingPdf		= 	logStartingPdf 		+ 	log(ss.beta.pdf( stateOld['m'][i]/20, a = alpha, b = beta))
		return logPropositionPdf - logStartingPdf

	def rvM(self, state):
		mNew = np.zeros(self.K)
		for i in range(self.K):
			if sum(state['Y'][i]) == 0:
				p = 0.0001
			else:
				nominator = sum([(x*y) for x,y in zip(state['X'],state['Y'][i])])
				denominator = 20*sum(state['Y'][i])
				p = nominator/denominator
			alpha = p/min(p,1.-p)
			beta  = (1.-p)/min(p,1.-p)

			mNew[i] = 20*ss.beta.rvs( a = alpha, b = beta )
		mNew = np.sort(mNew)
		return {'m':mNew}


proposal = Q(len(w))
print(Y)
stateOld = {'X':X, 'Y':Y, 'w': w, 'm':m}
stateNew = stateOld.copy()
stateNew.update(proposal.rvM(stateOld))
print(proposal.q(stateOld, stateNew))