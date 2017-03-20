from model import Gibbs, systematicUpdate, randomUpdate, generateSample
import pandas as pd
# from pprint import pprint as print	

stepsNo 	= 100000
hyper 		= { 'G': 2.,'D': 4.,'A': 2.,'B': 1. }
obsX, realP = generateSample(n=100, **hyper)

initialParams = {   'mu' :   obsX.mean(), 
                    'lam':   1./obsX.var()  }

X = {   'mean':obsX.mean(), 
        'var' :obsX.var(),
        'n'   :len(obsX)    }

res_systematicGibbs = Gibbs(X, initialParams, hyper, systematicUpdate, stepsNo)
res_systematicGibbs = pd.DataFrame(res_systematicGibbs)

res_randomGibbs = Gibbs(X, initialParams, hyper, randomUpdate, stepsNo)
res_randomGibbs = pd.DataFrame(res_randomGibbs)

pd.DataFrame([realP]).to_csv('realParameters.csv', index=False)
res_systematicGibbs.to_csv('systematicGibbs.csv', index=False)
res_randomGibbs.to_csv('randomGibbs.csv', index=False)