import numpy.random as npr 

class Gibbs:
    '''The Gibbs algorithm: a quasi virtual class.'''
    def __init__(self, model, state0, stepsNo, burnIn):
        self.stepsNo = stepsNo
        self.model   = model            #class: self.rv - functions, self.params - dict
        self.state   = state0           #dict: {'mu', 'lam'}
        self.burnIn  = burnIn

    def update(self):
        raise NotImplementedError
    
    def run(self):
        states = []
        collectData = False 
        for i in range(self.stepsNo):
            collectData = True if i >= self.burnIn else False
            self.update()
            if collectData:
                states.append(self.state.copy())
        return states

class GibbsRandomUpdate(Gibbs):
    '''The Gibbs algorithm with random update.'''
    def update(self):
        for m in npr.choice(self.model.params, 1, replace = False):
            self.state[m] = self.model.rv[m](**self.state)

class GibbsSystematicUpdate(Gibbs):
    '''The Gibbs algorithm with systematic update.'''
    def update(self):
        for m in self.model.params:
            self.state[m] = self.model.rv[m](**self.state)

def gibbs_algorithm(model, state0, update, stepsNo, burnIn=5000):
    '''A wrapper for calling the Gibbs algorithm.'''
    algo = {   
        'systematic':   GibbsSystematicUpdate,
        'random':       GibbsRandomUpdate 
    }[update](model, state0, stepsNo, burnIn)
    res = algo.run()
    return res
