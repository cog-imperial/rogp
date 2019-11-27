#!/usr/bin/env
"""core.py: build a robust GP-constrained pyomo model."""
import warnings
import numpy as np
import pyomo.environ as p
from . import kernels
from . import plot


class Standard():
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

    def predict_mu(self, x, z=None, cons=None):
        """ Predict mean from GP at x. """
        # Scale input
        x_norm = self.norm.x.normalize(x)
        # Calculate mean
        y = self._predict_mu(x_norm)
        # Unscale output
        return self.norm.y.inverse_mean(y)

    def _predict_cov(self, x):
        K_x_x = self.kern(x, x)
        K_x_X = self.kern(x, self.X)

        # Sig = K_x_x - K_x_X*W_inv*K_X_x
        SIG = K_x_x - np.matmul(np.matmul(K_x_X, self.woodbury_inv),
                                K_x_X.T)
        SIG += np.diag(np.repeat(self.likelihood_variance,
                                 x.shape[0]))
        return SIG

    def predict_cov(self, x):
        """ Predict covariance between two points from GP. """
        # Scale inputs
        x = self.norm.x.normalize(x)
        # Calculate covariance
        cov_norm = self._predict_cov(x)
        # Unscale output
        cov = self.norm.y.inverse_variance(cov_norm)
        return cov

    def predict(self, x):
        return self.predict_mu(x), self.predict_cov(x)

    def plot(self):
        plot.plot(self.gp)


class Warped(Standard):
    """
    Class for adding GP constraints from a GPy model to a Pyomo model.

    :param gp: GPy.models.gp_regression.WarpedGP object
    :param kern: type of kernel function to use
    :type kern: str

    """
    def _warp(self, y):
        """
        Transform y with warping function

        z = y*d + sum{a*tanh(b*(y + x))}
        """
        tanh = np.vectorize(p.tanh)
        d = self.gp.warping_function.d
        mpsi = self.gp.warping_function.psi

        z = d * y
        for i in range(len(mpsi)):
            a, b, c = mpsi[i]
            z += a * tanh(b * (y + c))
        return z

    def warp(self, y):
        # Scale input
        y_norm = self.norm.y.normalize(y)
        return self._warp(y_norm)

    def _warp_deriv(self, y):
        tanh = np.vectorize(p.tanh)
        d = self.gp.warping_function.d
        mpsi = self.gp.warping_function.psi

        S = (mpsi[:, 1] * (y[:, :, None] + mpsi[:, 2])).T
        R = tanh(S)
        D = 1 - (R ** 2)

        GRAD = (d + (mpsi[:, 0:1][:, :, None]
                     * mpsi[:, 1:2][:, :, None]
                     * D).sum(axis=0)).T

        return GRAD

    def warp_deriv(self, y):
        # Scale input
        y_norm = self.norm.y.normalize(y)
        return self._warp_deriv(y_norm)

    def predict_mu(self, x, z, cons):
        """ Predict mean from GP at x. """
        # Scale input
        x_norm = self.norm.x.normalize(x)
        # Calculate mean
        y = self._predict_mu(x_norm)
        # Scale in observation space
        z_norm = self.norm.y.normalize(z)
        # Set to prediction y in latent space
        diff = self.warp(z_norm) - y
        for d in np.nditer(diff, ['refs_ok']):
            cons.add(d.item() == 0)
        return z

    def predict_mu_latent(self, x):
        """ Predict mean from GP at x in latent space (normalized). """
        # Scale input
        x_norm = self.norm.x.normalize(x)
        # Calculate mean
        y = self._predict_mu(x_norm)
        return y

    def predict_cov_latent(self, x):
        """ Predict covariance between two points from GP. """
        # Scale inputs
        x = self.norm.x.normalize(x)
        # Calculate covariance
        cov = self._predict_cov(x)
        return cov

    def predict_latent(self, x):
        return self.predict_mu_latent(x), self.predict_cov_latent(x)

    def predict(self, x, z, cons):
        return self.predict_mu(x, z, cons)

    def predict_cov(self, x):
        warnings.warn("Cant predict variance in observation space."
                      " Predicting in latent space instead")
        return self.predict_cov_latent(x)
