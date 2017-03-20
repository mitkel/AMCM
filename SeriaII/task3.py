# Chciałem wykorzystać metodę diff z sympy, ale niestety
# okazało się, że w wyniku nie dostajemy wyrażenia, które
# można zamienić na funkcję lambda (nie działa metoda lambdify)

import sympy as sy
from mpmath import ln
import itertools as it
from sympy.matrices import hessian



def Hes():
	H_sy = {}
	H = {}

	x, w, m1, m2 = sy.symbols('x, w, m1, m2')
	f = -sy.ln(w*sy.exp(-0.5*(x-m1)**2) + (1-w)*sy.exp(-0.5*(x-m2)**2))
	theta = [w, m1, m2]
	for x,y in it.product( theta, theta ):
		H_sy[ str(x),str(y) ] = dummify_undefined_functions( \
			sy.diff(f, x, y))
		H[ str(x), str(y) ] = \
			sy.utilities.lambdify( (x,w,m1,m2), H_sy[str(x), str(y)])
		print( H[str(x), str(y)](0,0.3,-1,1) )
	return H_sy, H

# funkcja znaleziona w internecie, która ma mapoważ wyrażenie
# zawierające funkcje z numpy na napis symboliczny z sympy
def dummify_undefined_functions(expr):
    mapping = {}    

    # replace all Derivative terms
    for der in expr.atoms(sy.Derivative):
        f_name = der.expr.func.__name__
        var_names = [var.name for var in der.variables]
        name = "d%s_d%s" % (f_name, 'd'.join(var_names))
        mapping[der] = sy.Symbol(name)

    # replace undefined functions
    from sympy.core.function import AppliedUndef
    for f in expr.atoms(AppliedUndef):
        f_name = f.func.__name__
        mapping[f] = sy.Symbol(f_name)

    return expr.subs(mapping)


theta = ['w', 'm1', 'm2']
for x, y in it.product( theta, theta ):
	f, g = Hes()
	# print( f[x,y] )
	# print( g[x,y](0,0.3,-1,1) )