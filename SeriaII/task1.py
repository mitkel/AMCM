# generate the sample given THETA = (w, m1, m2) = (1/3, 0, 1)
# generate sample X_1, ... ,X_n ~ wN(m1, 1) + (1-w)N(m2,1)

import scipy.stats as ss
import numpy as np
from math import sqrt

params = {	'w': 0.01, 
			'm1': 0,
			'm2': 5}

def generate_sample(n, w, m1, m2):
	X = []
	for _ in range(n):
		s = np.random.choice(a=(m1,m2), p=[w,1.-w])
		X_n = ss.norm.rvs( loc=s, scale=1 )
		X.append(X_n)
	return X

# print(generate_sample(100, **params))