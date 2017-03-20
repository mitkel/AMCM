from MetropolisHastings import MetropolisHastings
import distributions as d
import scipy.stats as ss
import matplotlib.pyplot as plt

mc_steps = 10000
burnIn = 1000
alpha = 0.5

N = [d.Exp1, d.Exp2, d.Norm01, d.Norm11, d.Norm11]
T = [d.pseudo_target(t,d.target) for t in N]
Prop = d.Q(alpha)
state0 = 0

PMMH = [MetropolisHastings(mc_steps, burnIn, t, state0, Prop) for t in T]

for targ, alg in zip(N,PMMH):
	estimation_MH = [state0]
	for _, s in enumerate(alg.run()):
		if _ > alg.burnIn:
			estimation_MH.append(s)
	ss.probplot(estimation_MH, plot=plt)
	plt.savefig('Q' + targ.__name__ + '.png')
	plt.clf()
	plt.hist(estimation_MH)
	plt.savefig('Hist' + targ.__name__ + '.png')
	plt.clf()
