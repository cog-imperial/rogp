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
        x = np.ones(dim)
        y = np.zeros(dim)
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
        x = np.ones(dim)
        m = p.ConcreteModel()
        m.y = p.Var(range(dim))
        assert isinstance(kern.calc(list(m.y.values()), x),
                          UnaryFunctionExpression)
        assert isinstance(kern.calc(x, [y for y in m.y[:]]),
                          UnaryFunctionExpression)

    @pytest.mark.parametrize("dim", [
        1,
        3,
        5
    ])
    def test_rbf_calc_expression_array(self, dim):
        kern = rogp.kernels.RBF()
        X = np.ones(dim)
        m = p.ConcreteModel()
        m.Y = p.Var(range(dim))
        Y = np.array(list(m.Y.values()))
        assert isinstance(kern.calc(Y, X),
                          UnaryFunctionExpression)

    def test_rbf_calc_lengthscale(self):
        """ Make sure lengthscale is set correctly. """
        kern = rogp.kernels.RBF(lengthscale=2)
        x = np.ones(4)
        y = np.zeros(4)
        assert kern.calc(x, y) == math.exp(-0.5)

    def test_rbf_calc_variance(self):
        """ Make sure variance is set correctly. """
        kern = rogp.kernels.RBF(variance=2)
        x = -np.ones(1)
        y = np.zeros(1)
        assert kern.calc(x, y) == 2*math.exp(-0.5)


if __name__ == '__main__':
    pytest.main()
