import numpy as np
import numpy.random as npr
import scipy.stats as ss
import matplotlib.pyplot as plt

from task1 import draw_sample

def sis( particlesNo, Y, alpha, beta, mu, sigma ):
	X = []
	W = []

	X.append(npr.normal(loc = mu, scale = sigma, size = particlesNo))
	W.append(ss.norm.pdf(Y[0], scale = beta*np.exp(X[0]/2)))
	W[0] = W[0]/sum(W[0])

	for i in range(len(Y)-1):
		X.append(npr.normal(loc = alpha * X[i], scale = sigma))
		W.append(W[i]*ss.norm.pdf(Y[i+1], scale = beta*np.exp(X[i+1]/2)))
		W[i+1] = W[i+1]/sum(W[i+1])

	return X, W

params = \
	{
	'alpha':		.2,
	'beta':			.1,
	'mu':			.1,
	'sigma':		np.sqrt(.1),
	}
particlesNo = 100

for T in [5,10,30]:
	X_true, Y = draw_sample(T, **params)
	X, W 	= sis( particlesNo = particlesNo, Y = Y, **params )
	X_est 	= [sum(W[i]*X[i]) for i in range(T)]
	t = range(T)
	plt.plot(t, X_true, 'g', label = 'X_true')
	plt.plot(t, Y, 'm', label = 'Y')
	plt.plot(t, X_est, 'r', label = 'X_est')
	axes = plt.gca()
	axes.set_ylim([-2,2])
	plt.legend(loc = 1)
	plt.savefig('SIS_T' + str(T) + '.png')
	plt.clf()

	if T == 10:
		for n in [2,5,10]:
			plt.hist(W[n-1], bins = 30)
			plt.savefig('SIS_hist' + str(n) + '.png')
			plt.clf()