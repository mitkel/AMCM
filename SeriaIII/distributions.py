import scipy.stats as ss
import numpy.random as npr
import numpy as np
from math import log, exp

# klasa zawierająca rozkłady i gęstości proposala z częsci M-H naszego algorytmu
# znormalizowany wektor średnich będziemy losować z rozkładu produktowego beta() o średnich sum(X_nY_{k,n})/(sum(Y_{k,·}))
class Q():
	def __init__(self, K, sampleSize, lower, upper):
			self.K = K
			self.sampleSize = sampleSize
			self.lower = lower
			self.upper = upper
			self.range = self.upper - self.lower

	def q( self, stateNew, stateOld):
		logPropositionPdf 	= 0
		for i in range(self.K):
			if sum(stateOld['Y'][i]) > 0:

				# X należy zestandaryzować do przedziału (0,1), tj X_new = (X_old - lower)/range
				nominator = sum([((x-self.lower)*y) for x,y in zip(stateOld['X'],stateOld['Y'][i])])
				denominator = sum(stateOld['Y'][i])*(self.range)
				p = nominator/denominator
				alpha = self.range*p/min(p,1.-p)
				beta  = self.range*(1.-p)/min(p,1.-p)
				logPropositionPdf	=	logPropositionPdf 	+ 	log(ss.beta.pdf( (stateNew['m'][i]-self.lower)/self.range , a = alpha, b = beta))
		return logPropositionPdf

	def rv(self, state):
		mNew = np.zeros(self.K)
		for i in range(self.K):
			if sum(state['Y'][i]) == 0:
				p = 0
			else:
				nominator = sum([((x-self.lower)*y) for x,y in zip(state['X'],state['Y'][i])])
				denominator = sum(state['Y'][i])*self.range
				p = nominator/denominator
				alpha = self.range*p/min(p,1.-p)
				beta  = self.range*(1.-p)/min(p,1.-p)
			if p<=0 or p>=1:		
				mNew[i] = state['m'][i]
			else:
				mNew[i] = self.lower + (self.range)*ss.beta.rvs( a = alpha, b = beta )
			# mNew = np.sort(mNew)

		return {'m':mNew}

# zakładam prior z produktowego rozkładu jednostajnego U(0,20)
class target():
	def __init__(self, K, sampleSize):
		self.K = K
		self.sampleSize = sampleSize

	def logUnnormPDF(self, state):
	    s = 0
	    for i in range(self.K):
	        for j in range(self.sampleSize):
	            s = s-0.5*((state['X'][j]-state['m'][i])**2)*state['Y'][i][j] # + state['Y'][i][j]*log(state['w'][i]) - w naszym modelu ten fragment nie jest potrzebny, bo w_k nie będą update'owane w kroku MH
	    return s

# klasa zawierająca rozkłady warunkowe do części Gibbsowej naszego algorytmu
class G():
	def __init__(self, K, sampleSize):
		self.K = K
		self.sampleSize = sampleSize
		self.params = ['w', 'Y']
		self.rv 	= {'w': self.rv_w, 'Y': self.rv_Y}

	# okazuje się, że w zależy tylko od częstości Y oraz w_k | m,X,Y ~ Dir( 1+sum(Y_{1,i}), ... , 1+sum(Y_{K,i}) ).
	def rv_w(self, state):
		alpha = []
		[(alpha.append(1+sum(state['Y'][i]))) for i in range(self.K)]
		return npr.dirichlet(alpha)

	# zmienną Y rosujemy z rozkładu wielomianowego, gdyż p(Y_{k,n}|m,w,X) ~ [w_k*g_mk(X_n)]^(Y_{k,n}), czyli wektor Y_{·,n} ~ Multinom(w_1*g_m1(X_n), ..., w_K*g_mK(X_n))
	def rv_Y(self, state):
		Ynew = np.zeros(shape = (self.K, self.sampleSize) )
		for i in range(self.sampleSize):
			p = np.zeros(self.K)
			s = 0
			for j in range(self.K):
				p[j] = state['w'][j]*exp(-0.5*(state['X'][i]-state['m'][j])**2)
				s = s + p[j]
			p = p/s # normalizacja
			Ynew[npr.choice(self.K,p=p)][i] = 1
		return Ynew
