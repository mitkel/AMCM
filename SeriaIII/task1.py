import numpy as np
import numpy.random as npr

class mixture():
	def __init__(self, w, m):
		self.K = len(w)
		self.weights = w
		self.means = m

	def draw_sample(self, N):
		drawn_index = npr.choice(self.K, N, p=self.weights)
		sample = []
		Y = [[0]*N for i in range(self.K)]
		for i in range(N):
			sample.append(npr.normal(self.means[drawn_index[i]], 1))
			Y[drawn_index[i]][i]=1
		return sample, Y

def draw_means(K, spacing, lower, upper):
	start = npr.uniform(low = lower, high = upper - (lower+(K-1)*spacing))
	return list(np.arange( start, start+K*spacing, spacing ))

def draw_weights(K):
	return npr.dirichlet(np.ones(K))
