import sympy as sy
import itertools as it
from sympy.matrices import hessian

def Hes():
	x, w, m1, m2 = sy.symbols('x, w, m1, m2')
	f 		= -sy.log(w*sy.exp(-0.5*(x-m1)**2) + (1-w)*sy.exp(-0.5*(x-m2)**2))
	theta 	= [w, m1, m2]
	H_sy 	= hessian( f, theta )
	H = {}
	mapping = {'w':0, 'm1':1, 'm2':2}
	for i,j in it.product(theta, theta):
		H[str(i),str(j)] = sy.utilities.lambdify( (x,w,m1,m2), \
			H_sy[mapping[str(i)], mapping[str(j)]])
		# H[str(i),str(j)] = H_sy[mapping[str(i)], mapping[str(j)]]
	# H_sy = sy.utilities.lambdify( (x,w,m1,m2), H_sy)
	return H, H_sy

g = Hes()
print(g(1,0.3,-1,2))