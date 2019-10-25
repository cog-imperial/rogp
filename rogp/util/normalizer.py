#!/usr/bin/env
import GPy


class Normalizer:
    """
    Convenience class which wraps two normalizers, one for inputs and one for
    outputs

    Args:
        X_norm: normalizer for inputs
        Y_norm: normalizer for outputs

        both default to GPy.util.normalizer.Standardize()

    """
    def __init__(self, X_norm=None, Y_norm=None):
        # Default normalizers
        if X_norm is None:
            X_norm = GPy.util.normalizer.Standardize()
        if Y_norm is None:
            Y_norm = GPy.util.normalizer.Standardize()
        self.X_norm = X_norm
        self.Y_norm = Y_norm

    def scale_by(self, X, Y):
        self.X_norm.scale_by(X)
        self.Y_norm.scale_by(Y)

    def normalize(self, X, Y):
        return self.X_norm.normalize(X), self.Y_norm.normalize(Y)


class IdentityNorm():
    def __init__():
        pass

    def normalize(self, X):
        return X

    def inverse_mean(self, Y):
        return Y

    def inverse_cov(self, Y):
        return Y
