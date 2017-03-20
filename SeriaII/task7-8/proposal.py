import scipy.stats as ss
import numpy as np
from math import sqrt, log

class Q():
	# rozkład beta o zadanej variancji nie jest chyba najlepszym jądrem przejścia -
	# VarB in min(EB, 1-EB), zatem jeśli przypadkowo wybierzemy zbyt mały, albo zbyt
	# duży parametr w, to może się okazać, że szukany rozkład beta po prostu nie istnieje
	# przy założeniu stałej wariancji, dlatego też postanowiłem unormować rozkład
	# propozycji i założyć a = w, b = 1-w, wtedy Ex(B)=w; Var(B)=(1-w)w

	# update: a = w i b = 1-w nadal sprawiały problemy - w związku z dużą gęstością na
	# brzegach rozkład szybko uciekał do 0 lub 1. Poprawiłem to normalizując oba parametry
	# przez min(w,1-w)**2 (lub po prostu min(w,1-w)) - skalowanie parametrów nie wpływa 
	# na średnią, a jedynie zmienia wariancję. Ponadto im większe a i b, tym rozkład jest 
	# bardziej scentrowany i losowanie nie sprawia problemów.
	def __init__(self, st_dev_beta=.1, sigma2=1):
		self.st_dev_beta = st_dev_beta
		self.sigma2 = sigma2

	def q(self,w,m1,m2,v,N1,N2):
		normal_part = 	ss.norm.pdf(N1, loc = m1, scale = sqrt(self.sigma2)) \
					*	ss.norm.pdf(N2, loc = m2, scale = sqrt(self.sigma2)) \
					+	ss.norm.pdf(N2, loc = m1, scale = sqrt(self.sigma2)) \
					*	ss.norm.pdf(N1, loc = m2, scale = sqrt(self.sigma2))

		alpha = w/min(w,1.-w)
		beta  = (1.-w)/min(w,1.-w)
		beta_part	=	ss.beta.pdf( v, a = alpha, b = beta)
		if normal_part == 0 or beta_part == 0:
			return -np.inf
		else:
			return log(normal_part)+log(beta_part)


	def rv(self,w,m1,m2):
		Z1 = ss.norm.rvs( loc=m1, scale=sqrt(self.sigma2))
		Z2 = ss.norm.rvs( loc=m2, scale=sqrt(self.sigma2))

		# max dodany, żeby nie wyjść poza dziedzinę
		# alpha 	= max(w*(-1 + (w**2)*(1-w)/self.st_dev_beta**2),.01)
		# beta 	= max(alpha*(1-w)/w,.01)
		alpha = w/min(w,1.-w)
		beta  = (1.-w)/min(w,1.-w)
		v 		= ss.beta.rvs( a = alpha, b = beta )

		return {'w':v, 'm1':min(Z1,Z2), 'm2':max(Z1,Z2)}

# params = (0,1,.3)
# Q = Q()
# for _ in range(100):
# 	print(Q.rv(*params))