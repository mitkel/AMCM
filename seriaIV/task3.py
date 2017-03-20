import numpy as np
import numpy.random as npr
import scipy.stats as ss
import matplotlib.pyplot as plt

from task1 import draw_sample


def sir(particlesNo, Y, alpha, beta, mu, sigma):
	X = []

	x = npr.normal(loc=mu, scale=sigma, size=particlesNo)
	w = ss.norm.pdf(Y[0], scale = beta*np.exp(x/2))
	X.append(npr.choice(x,particlesNo,True,w/sum(w)))

	for i in range(len(Y) - 1):
		x = npr.normal(loc=alpha * X[i], scale=sigma)
		w = ss.norm.pdf(Y[i + 1], scale=beta * np.exp(x / 2))
		X.append(npr.choice(x,particlesNo,True,w/sum(w)))
	return X

params = \
	{
		'alpha': .2,
		'beta': .1,
		'mu': .1,
		'sigma': np.sqrt(.1),
	}
particlesNo = 100

for T in [5, 10, 30]:
	X_true, Y = draw_sample(T, **params)
	X = sir(particlesNo=particlesNo, Y=Y, **params)
	X_est = [sum(X[i])/particlesNo for i in range(T)]
	t = range(T)
	plt.plot(t, X_true, 'g', label='X_true')
	plt.plot(t, Y, 'm', label='Y')
	plt.plot(t, X_est, 'r', label='X_est')
	axes = plt.gca()
	axes.set_ylim([-2, 2])
	plt.legend(loc=1)
	plt.savefig('SIR_T' + str(T) + '.png')
	plt.clf()
