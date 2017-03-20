import sympy as sy
import numpy as np
from sympy.matrices import hessian
import scipy.stats as ss
from math import sqrt, log

class target:
	def __init__(self, mc_steps, sample=[], sigma2=1):
		self.sample     = sample
		self.N          = len(sample)
		self.mc_steps   = mc_steps
		self.sigma2     = sigma2

		x, w, m1, m2 = sy.symbols('x, w, m1, m2')
		f       = w * sy.exp( - (x-m1)**2/2 ) + (1-w) * sy.exp( - (x-m2)**2/2 )
		theta = [w, m1, m2]
		self.d  = len(theta)
		H       = -hessian( sy.log(f), theta )
		self.hessian = sy.utilities.lambdify( (x,w,m1,m2), H )
		self.f       = sy.utilities.lambdify( (x,w,m1,m2), sy.log(f) )

	def rv(self, w, m1, m2):
		W = np.random.choice(a=(m1, m2), p=[w,1.-w], size=1)
		X = ss.norm.rvs( loc = W, scale = sqrt( self.sigma2))
		return X

	def rvs(self, N, w, m1, m2):
		for n in range(N):
			X = self.rv(w, m1, m2)
			yield X

	def JeffreysPriors(self, w, m1, m2):
		FM = np.zeros(( self.d, self.d ))
		#tutaj szacujemy pewną całkę, więc należało podmienić self.N (wielkość próbki)
		# na self.mc_steps. Ogólnie moim zdaniem self.N jest tutaj niepotrzebne. Można
		# nawet go nie uwzględniać przy liczeniu informacji Fishera, która spełnia
		# I = nI_1, ponieważ mamy ją policzyć tylko z dokładnośćią do stałej
		for i, X in enumerate( self.rvs(self.mc_steps, w, m1, m2) ):
			FM  = FM*i/(i+1.) + np.array(self.hessian(X, w, m1, m2))/(i+1.)
		return sqrt( abs(np.linalg.det(FM)) )

	def logUnnormPDF1(self, w, m1, m2):
		logF = 0
		for x in self.sample:
			logF = logF + self.f(x, w, m1, m2)
		return logF

	def logUnnormPDF2(self, w, m1, m2):
		return log(self.JeffreysPriors(w, m1, m2))


	def logUnnormPDF(self, w, m1, m2):
		logF = self.logUnnormPDF1(w, m1, m2) + self.logUnnormPDF2(w, m1, m2)
		return logF

# sample = [1,2]
# state = (0,1,.3)
# T = target(20, sample)
# print(T.logUnnormPDF(state))