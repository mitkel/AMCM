from simpleHierarchicalModel import simpleHierarchicalModel, sample
import pandas as pd
from pprint import pprint as print  
from gibbs import gibbs_algorithm
# import pudb; pu.db

hyper       = { 'G': 2.,'D': 4.,'A': 2.,'B': 1. }
obsX, realP = sample(n=20, **hyper)

X = {   'mean':obsX.mean(), 
        'var' :obsX.var(),
        'n'   :len(obsX)    }

model = simpleHierarchicalModel(obsX, **hyper)

state0 = {  'mu' :   obsX.mean(), 
            'lam':   1./obsX.var()  }

stepsNo = 10000
burnIn  = 1000

gibbs_r = gibbs_algorithm(model, state0, 'random', stepsNo)
gibbs_r = pd.DataFrame(gibbs_r)
# gibbs_r.to_csv('randomGibbs.csv', index=False)
gibbs_s = gibbs_algorithm(model, state0, 'systematic', stepsNo)
gibbs_s = pd.DataFrame(gibbs_s)
# gibbs_s.to_csv('systematicGibbs.csv', index=False)

print(realP)
print(gibbs_r)
print(gibbs_s)