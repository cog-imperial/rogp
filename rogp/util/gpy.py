#!/usr/bin/env
from .. import core


def from_gpy(gp, kern='RBF', norm=None):
    if hasattr(gp, 'warping_function'):
        return core.Warped(gp, kern=kern, norm=norm)
    else:
        return core.Standard(gp, kern=kern, norm=norm)
