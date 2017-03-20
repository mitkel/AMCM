from MetropolisHastings import MetropolisHastings
from task2_distributions import draw_sample, Q, pseudo_target

import matplotlib.pyplot as plt
import numpy as np

mc_steps = 1000
burnIn = 100
sample_size = 10
space = [-1, 1, 2, 4]
K = 10
theta0 = 0

theta_true, x, Y = draw_sample(sample_size, space)
Prop = Q(space)
pseudos = [pseudo_target(space, 1, Y), pseudo_target(space, K, Y)]
PMMHs = [MetropolisHastings(mc_steps, burnIn, pseudo, theta0, Prop) for pseudo in pseudos]

print("True theta: " + str(theta_true))
for targ, alg in zip(pseudos, PMMHs):
	estimation_MH = [theta0]
	for _, s in enumerate(alg.run()):
		if _ > alg.burnIn:
			estimation_MH.append(s)
	plt.hist(estimation_MH)
	plt.savefig('task2_Hist_' + str(targ.K) + '.png')
	plt.clf()
	print("Theta estimated with " +str(targ.K) + "-clusters: " + str(np.mean(estimation_MH)))