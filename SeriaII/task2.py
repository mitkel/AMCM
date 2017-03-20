import scipy.stats as ss
from math import sqrt

class Q():
	def __init__(self, st_dev_beta, sigma2):
		self.st_dev_beta = st_dev_beta
		self.sigma2 = sigma2

	def q(m1,m2,w,N1,N2,v):
		normal_part = 	ss.norm.pdf(N1, loc = m1, scale = sqrt(self.sigma2)) \
					*	ss.norm.pdf(N2, loc = m2, scale = sqrt(self.sigma2)) \
					+	ss.norm.pdf(N2, loc = m1, scale = sqrt(self.sigma2)) \
					*	ss.norm.pdf(N1, loc = m2, scale = sqrt(self.sigma2))
		normal_part = normal_part/2
		
	# w 	= Ex(B) = alpha/(alpha+beta)
	# dev2 	= Var(B) = alphabeta/[(alpha+beta)2(alpha+beta+1)]
		alpha 		= 	w*(-1 + (w**2)*(1-w)/self.st_dev_beta)
		beta 		= 	alpha*(1-w)/2
		beta_part	=	ss.beta.pdf( v, a = alpha, b = beta)

		return normal_part*beta_part


	def rv( m1,m2,w ):
		Z1 = ss.norm.rvs( loc=m1, scale=sqrt(self.sigma2), size=n )
		Z2 = ss.norm.rvs( loc=m2, scale=sqrt(self.sigma2), size=n )

		alpha 	= w*(-1 + (w**2)*(1-w)/self.st_dev_beta)
		beta 	= alpha*(1-w)/2
		v 		= ss.beta( a = alpha, b = beta )

		return min(Z1,Z2), max(Z1,Z2), v