#!/usr/bin/env
"""kernels.py: kernel functions which can take Pyomo variables as arguments."""
import pyomo.environ as p
import numpy as np


class Kernel():
    """ Abstract kernel. """
    def __init__(self):
        raise NotImplementedError

    def calc(self, x1, x2):
        raise NotImplementedError

    def __call__(self, x, y):
        return self.calc(x, y)

    def euclidian_squared(self, x, y):
        """ Vectorized euclidian squared distance between X and Y.

            :param x: numpy.ndarray (n,k)
            :param y: numpy.ndarray (m,k)

            returns numpy.ndarray (n, m)
        """
        assert x.shape[1] == x.shape[1]
        n = x.shape[0]
        m = y.shape[0]
        X = x[:, None, :].repeat(m, axis=1)
        Y = y[None, :, :].repeat(n, axis=0)
        return np.sum((X - Y)**2, axis=2)


class RBF(Kernel):
    """ Squared exponential kernel function. """
    def __init__(self, lengthscale=1.0, variance=1.0):
        assert lengthscale > 0
        assert variance > 0
        self.lengthscale = lengthscale
        self.variance = variance

    def calc(self, x, y):
        """ Calculate k(x1, x2)

        :param x:
        :param y:
        :type x: int, float, pyomo.environ.Var, ...
        :type y: int, float, pyomo.environ.Var, ...
        :returns: squared exponential distance
        """
        r_sqr = self.euclidian_squared(x, y)/(self.lengthscale**2)
        exp = np.vectorize(p.exp)
        return self.variance * exp(-0.5 * r_sqr)
