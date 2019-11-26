#!/usr/bin/env
import GPy


class Normalizer:
    """
    Convenience class which wraps two normalizers, one for inputs and one for
    outputs

    Args:
        x: normalizer for inputs
        y: normalizer for outputs

        both default to GPy.util.normalizer.Standardize()

    """
    def __init__(self, x=None, y=None):
        # Default normalizers
        if x is None:
            x = GPy.util.normalizer.Standardize()
        if y is None:
            y = GPy.util.normalizer.Standardize()
        self.x = x
        self.y = y

    def scale_by(self, X, Y):
        self.x.scale_by(X)
        self.y.scale_by(Y)

    def normalize(self, X, Y):
        return self.x.normalize(X), self.y.normalize(Y)


class IdentityNorm():
    def __init__():
        pass

    def normalize(self, X):
        return X

    def inverse_mean(self, Y):
        return Y

    def inverse_cov(self, Y):
        return Y
