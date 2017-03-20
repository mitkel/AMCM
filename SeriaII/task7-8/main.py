from MetropolisHastings import MetropolisHastings
from delayedAcceptance import DelayedAcceptance
from proposal import Q
from target import target
from pprint import pprint as print
from timeit import default_timer as timer

N   = 40			# wielkość zaobserwowanej próbki
mc_steps_t = 40 	# liczba kroków wykorzystywana przy przybliżaniu Jeff-priora
mc_steps = 200    	# liczba kroków stosowana w MH i DA
burnIn = 50 		# liczba kroków pominiętych w obu algorytmach
levels = 2			# liczba poziomów w DA
params = { 'w':0.2, 'm1':0, 'm2':5 }	# prawdziwe parametry
state0 = { 'w':0.4, 'm1':0, 'm2':3 }	# stan startowy

T       = target(mc_steps_t)
sample  = list( T.rvs(N, **params) )

# musimy zmienić target, bo teraz naszym targetem jest rozkład thety po warunkiem
# wylosowanej próbki
Targ = target(mc_steps_t, sample)
Prop = Q()
MH = MetropolisHastings(mc_steps, burnIn, Targ, state0, Prop)
DA = DelayedAcceptance(mc_steps, burnIn, Targ, state0, Prop, levels)

print("Prawdziwe parametry: " + str(params))
for alg in [MH, DA]:
	start_timer = timer()
	estimation = state0
	for i, s in enumerate(alg.run()):
		if i > alg.burnIn:
			j = i - burnIn
			estimation = {k: estimation[k]*j/(j+1.)+s[k]/(j+1.) for k in estimation.keys()}
	end_timer = timer()
	print(alg.Name())
	print("Estimated: " + str(estimation))
	print(alg.Name() + " elapsed in " + str(end_timer-start_timer) + "s.")