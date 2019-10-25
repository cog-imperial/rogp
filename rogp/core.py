#!/usr/bin/env
"""core.py: build a robust GP-constrained pyomo model."""
import numpy as np
from . import kernels
from . import plot


class ROGP():
    """
    Class for adding GP constraints from a GPy model to a Pyomo model.

    :param gp: GPy.models.gp_regression.GPRegression object
    :param kern: type of kernel function to use
    :type kern: str

    """
    def __init__(self, gp, kern='RBF', norm=None):
        self.gp = gp
        self.norm = norm
        self.woodbury_vector = gp.posterior.woodbury_vector
        self.woodbury_inv = gp.posterior.woodbury_inv
        try:
            self.kern = getattr(kernels, kern)(gp.kern.lengthscale[0],
                                               gp.kern.variance[0])
        except NotImplementedError:
            print('Kernel ' + kern + ' is not implemented')
        self.likelihood_variance = self.gp.likelihood.variance[0]
        self.X = gp._predictive_variable
        self.N = len(self.X)

    def _predict_mu(self, x):
        """ Predict mean from GP at x. """
        K_x_X = self.kern.calc(x, self.X)
        mu = np.matmul(K_x_X, self.woodbury_vector)
        return mu

    def predict_mu(self, x):
        """ Predict mean from GP at x. """
        # Scale input
        x_norm = self.norm.X_norm.normalize(x)
        # Calculate mean
        y_norm = self._predict_mu(x_norm)
        # Unscale output
        y = self.norm.Y_norm.inverse_mean(y_norm)
        return y

    def _predict_cov(self, x):
        K_x_x = self.kern(x, x)
        K_x_X = self.kern(x, self.X)

        # Sig = K_x_x - K_x_X*W_inv*K_X_x
        return K_x_x - np.matmul(np.matmul(K_x_X, self.woodbury_inv),
                                 K_x_X.T)

    def predict_cov(self, x):
        """ Predict covariance between two points from GP. """
        # Scale inputs
        x = self.norm.X_norm.normalize(x)
        # Calculate covariance
        cov_norm = self._predict_cov(x)
        # Unscale output
        cov = self.norm.Y_norm.inverse_variance(cov_norm)
        return cov

    def predict(self, x):
        return self.predict_mu(x), self.predict_cov(x)

    def plot(self):
        plot.plot(self.gp)
