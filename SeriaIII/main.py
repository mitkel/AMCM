from MH_within_Gibbs import MH_within_Gibbs_RS, MH_within_Gibbs_DS
from distributions import Q, target, G
from task1 import mixture, draw_means, draw_weights

from pprint import pprint as print
from timeit import default_timer as timer
import numpy as np
from numpy.linalg import norm
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

# initial params
K 			= 2  # liczba klastrow
d 			= 5  # odstepy miedzy modami
lower 		= 0  # dolna granica na mody
upper 		= 20 # gorna granica na mody
sample_size = 50
mc_steps 	= 200
burnIn 		= 100

# drawing other initial params
m_true = draw_means(K, d, lower, upper)
w_true = draw_weights(K)

# drawing sample
true_distr = mixture(w_true, m_true)
X, Y_true = true_distr.draw_sample(sample_size)

# other params
Targ 	= target(K, sample_size)
Prop 	= Q(K, sample_size, lower, upper)
Gibbs 	= G(K, sample_size)
gParams = ['w', 'Y']	# Gibbs params
mParams = ['m']			# MH params

# drawing initial state
m = draw_means(K, d, lower, upper)
w = draw_weights(K)
Y = np.zeros(shape = (K, sample_size) )
for i in range(len(X)):
	Y[npr.choice(K)][i] = 1
Y_almost_true = Gibbs.rv_Y({'X':X, 'w':w_true, 'm':m_true})


state0 = {'X':X, 'Y':Y, 'w':w, 'm':m}

RS = MH_within_Gibbs_RS(mc_steps, burnIn, state0, Targ, Prop, Gibbs, mParams, gParams)
DS = MH_within_Gibbs_DS(mc_steps, burnIn, state0, Targ, Prop, Gibbs, mParams, gParams)

start_RS = timer()
w_estimation_RS = np.zeros(K)
m_estimation_RS = np.zeros(K)
for i, s in enumerate(RS.run()):
	if i > RS.burnIn:
		j = i - burnIn
		w_estimation_RS = [a*j/(j+1.) + b/(j+1.) for a,b in zip(w_estimation_RS, s['w'])]
		m_estimation_RS = [a*j/(j+1.) + b/(j+1.) for a,b in zip(m_estimation_RS, s['m'])]
end_RS = timer()


start_DS = timer()
w_estimation_DS = np.zeros(K)
m_estimation_DS = np.zeros(K)
for i, s in enumerate(DS.run()):	
	if i > DS.burnIn:
		j = i - burnIn
		w_estimation_DS = [a*j/(j+1.) + b/(j+1.) for a,b in zip(w_estimation_DS, s['w'])]
		m_estimation_DS = [a*j/(j+1.) + b/(j+1.) for a,b in zip(m_estimation_DS, s['m'])]
end_DS = timer()

# sortowanie wynikow względem mod
w_estimation_RS = [x for (y,x) in sorted(zip(m_estimation_RS,w_estimation_RS))]
m_estimation_RS = list(np.sort(m_estimation_RS))

w_estimation_DS = [x for (y,x) in sorted(zip(m_estimation_DS,w_estimation_DS))]
m_estimation_DS = list(np.sort(m_estimation_DS))

print("True w: " + str(w_true))
print("True m: " + str(m_true))
print("RS:")
print("Estimated w: " + str(w_estimation_RS))
print("Estimated m: " + str(m_estimation_RS))
print("DS:")
print("Estimated w: " + str(w_estimation_DS))
print("Estimated m: " + str(m_estimation_DS))

MLE_m = [sum(x*y for x,y in zip(X,Y_true[i]))/(sum(y for y in Y_true[i])) for i in range(K)]
MLE_w = [sum(Y_true[i])/sample_size for i in range(K)]
print("MLE of w:" + str(MLE_w))
print("MLE of m:" + str(MLE_m))

sns.distplot(X, hist=True, rug=True)
plt.show()
