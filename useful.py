import theano.tensor as T
from theano import shared
import numpy as np
from theano import function


class hiddenLayer():
    """ Hidden Layer class
    """
    def __init__(self, inputs, n_in, n_out, act_func):
        rng = np.random
        self.W = shared(np.asarray(rng.uniform(low=-4*np.sqrt(6. / (n_in + n_out)),
                                               high=4*np.sqrt(6. / (n_in + n_out)),
                                               size=(n_in, n_out)),
                                   dtype=T.config.floatX),
                        name='W')
        self.inputs = inputs
        self.b = shared(np.zeros(n_out, dtype=T.config.floatX), name='b')
        self.x = T.dvector('x')
        self.z = T.dot(self.x, self.W) + self.b
        self.ac = function([self.x], self.z)

a = hiddenLayer(np.asarray([1, 2, 3], dtype=T.config.floatX), 3, 3, T.tanh)
print a.ac(a.inputs)