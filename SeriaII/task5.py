import sympy as sy
from sympy.matrices import hessian
import numpy as np
from task1 import generate_sample
from math import sqrt

def Hes():
	x, w, m1, m2 = sy.symbols('x, w, m1, m2')
	f 		= -sy.log(w*sy.exp(-0.5*(x-m1)**2) + (1-w)*sy.exp(-0.5*(x-m2)**2))
	theta 	= [w, m1, m2]
	H_sy 	= hessian( f, theta )
	H_sy = sy.utilities.lambdify( (x,w,m1,m2), H_sy)
	return H_sy

def calculate_Fisher(n, w, m1, m2):
	sample = generate_sample(n, w, m1, m2)
	F = np.zeros((3, 3))
	for i in range(n):
		F = F*i/(i+1.)+Hes()(sample[i], w, m1, m2)/(i+1.)
	return sqrt(np.abs(np.linalg.det(F)))

# stepsNo	= 100
# sample_size = 20
# hyper 	= {'w':0.3, 'm1':1, 'm2':2}
# sample 	= generate_sample(sample_size, **hyper)

# FM = calculate_Fisher(stepsNo, **hyper)
# print(FM)