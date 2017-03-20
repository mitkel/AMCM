import numpy.random as npr
from math import exp, sqrt
import matplotlib.pyplot as plt

def draw_sample(T, alpha, beta, mu, sigma):
	sample_x = []
	sample_y = []
	sample_x.append( npr.normal(loc = mu, scale = sigma) )
	sample_y.append( npr.normal( loc = 0, scale = beta*exp(sample_x[0]/2) ) )
	for i in range(T-1):
		x = alpha*sample_x[i]+sigma*npr.normal()
		sample_x.append(x)
		y = npr.normal()*beta*exp(x/2)
		sample_y.append(y)

	return sample_x, sample_y


params = \
	{
	'alpha':	.2,
	'beta':		.1,
	'mu':		.1,
	'sigma':	sqrt(.1) 
	}

for T in [5,10,30]:
	X, Y = draw_sample(T, **params)
	t = range(T)
	plt.plot(t, X, 'g', label = 'X')
	plt.plot(t, Y, 'm', label = 'Y')
	plt.legend(loc = 1)
	plt.savefig('taks1_T' + str(T) + '.png')
	plt.clf()
