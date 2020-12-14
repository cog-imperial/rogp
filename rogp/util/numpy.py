#!/usr/bin/env
import numpy as np
import pyomo.environ as p


# TODO: Generalize to higher dimension
def _to_np_obj_array(x):
    """ Convert nested list to numpy array with dtype=object """
    X = np.empty((len(x), len(x[0])), dtype=object)
    X[:] = x
    return X


def _eval(x, evaluate):
    if evaluate:
        return x()
    else:
        return x


def pyomo_to_np(X, ind=None, evaluate=False):
    return _pyomo_to_np(X, ind=ind, evaluate=evaluate)


def _pyomo_to_np(X, ind=None, evaluate=False):
    if ind is None:
        XX = [[_eval(x, evaluate)] for _, x in X.items()]
    else:
        XX = [[_eval(X[i], evaluate)] for i in ind]
    if evaluate:
        return np.array(XX).astype('float')
    else:
        return _to_np_obj_array(XX)


def _eval_p(X, ind=None):
    if ind is None:
        return [x() for _, x in X.items()]
    else:
        return [X[i]() for i in ind]


def tanh(x):
    return (p.exp(2*x) - 1) / (p.exp(2*x) + 1)
