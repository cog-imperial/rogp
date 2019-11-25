#!/usr/bin/env
import pytest
import numpy as np
import math
import rogp.kernels
import pyomo.environ as p
from pyomo.core.expr.numeric_expr import UnaryFunctionExpression


class TestRBF:

    @pytest.mark.parametrize("dim", [
        1,
        2,
        10
    ])
    def test_rbf_calc_numeric(self, dim):
        """ Make sure result is correct when input is numerical values. """
        kern = rogp.kernels.RBF()
        x = np.ones((1, dim))
        y = np.zeros((1, dim))
        assert kern.calc(x, y) == math.exp(-0.5*dim)

    @pytest.mark.parametrize("dim", [
        1,
        3,
        5
    ])
    def test_rbf_calc_expression(self, dim):
        """ Make sure pyomo expression is returned when input is pyomo
        variables.

        """
        kern = rogp.kernels.RBF()
        x = np.ones((1, dim))
        m = p.ConcreteModel()
        m.y = p.Var(range(dim))
        y = np.empty((1, dim), dtype=object)
        y[0,:] = list(m.y.values())
        assert isinstance(kern.calc(x, y)[0, 0],
                          UnaryFunctionExpression)

    def test_rbf_calc_lengthscale(self):
        """ Make sure lengthscale is set correctly. """
        kern = rogp.kernels.RBF(lengthscale=2)
        x = np.ones((1, 4))
        y = np.zeros((1, 4))
        assert kern.calc(x, y) == math.exp(-0.5)

    def test_rbf_calc_variance(self):
        """ Make sure variance is set correctly. """
        kern = rogp.kernels.RBF(variance=2)
        x = -np.ones((1, 1))
        y = np.zeros((1, 1))
        assert kern.calc(x, y) == 2*math.exp(-0.5)


if __name__ == '__main__':
    pytest.main()
