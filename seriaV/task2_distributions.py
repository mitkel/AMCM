import numpy.random as npr
import numpy as np
import scipy.stats as ss


def draw_X(theta, space):
    weights = np.exp(20 * np.array(space) * theta)
    # jesli wylosowana theta jest zbyt duza, to wagi eksploduja
    if np.any(np.isinf(weights)):
        weights = np.array([0] * len(space))
        weights[space == np.max(np.array(space) * np.sign(theta))] = 1
    else:
        weights = weights / sum(weights)
    return npr.choice(a=space, replace=True, p=weights)


def draw_sample(sample_size, space):
    theta = npr.uniform(-20, 20)
    x = draw_X(theta, space)
    Y = npr.normal(loc=x + theta, scale=1, size=sample_size)
    return theta, x, Y


# nowa theta jest losowana z rozkladu jednostajnego u(-20,20), a nowy x z rozkladu warunkowego x|theta
class Q():
    def __init__(self, space):
        self.space = space

    def q(self, stateNew, stateOld):
        # # tym razem jadro przejsia nie jest symetryczne - jest za to niezalezne od poprzedniego stanu
        # weights = np.exp(20 * np.array(self.space) * stateNew['theta'])
        # if np.any(np.isinf(weights)):
        #     weights = np.array([0] * len(self.space))
        #     weights[self.space == np.max(np.array(self.space) * np.sign(stateNew['theta']))] = 1
        # else:
        #     weights = weights / sum(weights)
        # return weights[np.array(self.space) == stateNew['x']]
        return 1

    def rv(self, state):
        return npr.uniform(-20, 20)

    # 	thetaNew = npr.uniform(-20, 20)
    # 	x_new = draw_X(thetaNew, self.space)
    # 	return {'x':x_new, 'theta':thetaNew}


class pseudo_target():
    def __init__(self, space, K, obs):
        self.space  = space
        self.K      = K
        self.obs    = obs

    def UnnormPDF(self, state):
        UnnormPDF = 0
        for _ in range(self.K):
            x = draw_X(state, self.space)
            UnnormPDF = UnnormPDF + np.product(ss.norm.pdf(self.obs, loc = x + state))
        UnnormPDF = UnnormPDF/self.K
        return UnnormPDF