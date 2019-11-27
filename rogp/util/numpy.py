#!/usr/bin/env
import numpy as np


# TODO: Generalize to higher dimension
def _to_np_obj_array(x):
    """ Convert nested list to numpy array with dtype=object """
    X = np.empty((len(x), len(x[0])), dtype=object)
    X[:] = x
    return X


def _pyomo_to_np(X, ind=None):
    if ind is None:
        XX = [[x] for _, x in X.items()]
    else:
        XX = [[X[i]] for i in ind]
    return _to_np_obj_array(XX)


def _eval_p(X, ind=None):
    if ind is None:
        return [x() for _, x in X.items()]
    else:
        return [X[i]() for i in ind]
