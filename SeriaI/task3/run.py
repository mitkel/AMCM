from simpleHierarchicalModel import hierarchy_conditional, hierarchy_full
import pandas as pd
from pprint import pprint  
from math import exp, fsum, log
import numpy as np

hyper = { 'G': 2.,'D': 4.,'A': 2.,'B': 1. }

importance 	= hierarchy_full( n=20, **hyper )
realP, obsX = importance.rvState_and_observations()

# We could use a different hyper. 
# In real life we would definitely do that.
nominal = hierarchy_conditional(obsX, **hyper)

stepsNo = 10000
states 	= []
logWeights = []

# Here we shall make the simulation: the problem seems too easy to derive a separate class for algorithm. 
for _ in range(stepsNo):
	state = importance.rvState()
	states.append(state)	
	logP = nominal.unnorm_log_pdf(**state)
	logQ = importance.unnorm_log_pdf(**state)
	logWeights.append(logP-logQ)
		
# fsum claims to return a more accurate floating point number than sum.

logSumOfWeights = log(fsum( exp(lW) for lW in logWeights))
logWeights 		= [ lW-logSumOfWeights for lW in logWeights ]

def save(thing, path):
	pd.DataFrame(thing).to_csv(path, index=False)

save(states, 'importanceSampling.csv')
save(logWeights, 'logWeights.csv')
save([realP], 'realP.csv')
