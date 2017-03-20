from distributions import multivariate_t_student, multivariate_normal
from Importance import IS
import pandas as pd
from pprint import pprint  
from math import exp, fsum, log
import numpy as np

dim = 10
hyper1 = { 'df': 5., 'N':dim }
hyper2 = { 'mean_mtr':[0]*dim, 'dev_mtr':np.identity(dim), 'N':dim }

importance 	= multivariate_t_student( **hyper1 )
nominal 	= multivariate_normal( **hyper2 )
Algorithm = IS(stepsNo = 500, burnIn = 50, state0 = [0]*dim, nominal = nominal, importance = importance)

states 	= []
logWeights = []
logWeightsImportance = []
logWeightsNominal = []
Algorithm.run()


logSumOfWeights = log(fsum( exp(lW) for lW in Algorithm.weight_history_importance))
# logWeights 		= [ lW-logSumOfWeights for lW in Algorithm.weight_history ] 
# w kodzie uwzględniłem stałe normujące, więc powyższy fragment można pominąć
# dzięki temu łatwiej 'na oko' zobaczyć, czy stałe są w porządku
logWeights = Algorithm.weight_history
states 			= Algorithm.state_history
logWeightsNominal =  Algorithm.weight_history_nominal
logWeightsImportance = Algorithm.weight_history_importance

def save(thing, path):
	pd.DataFrame(thing).to_csv(path, index=False)

save(states, 'importanceSampling.csv')
save(logWeights, 'logWeights.csv')
save(logWeightsNominal, 'logWeightsNominal.csv')
save(logWeightsImportance, 'logWeightsImportance.csv')