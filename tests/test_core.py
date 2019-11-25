#!/usr/bin/env
import pytest
import numpy as np


@pytest.fixture

def rogp(scope='module'):
    import json
    import GPy
    import rogp

    with open('tests/data_warped_gp.json', 'r') as f:
        data = json.load(f)

    X = np.array(data['X'])[:, None]
    Y = np.array(data['Y'])[:, None]
    norm = rogp.util.Normalizer()
    norm.scale_by(X, Y)
    X, Y = norm.normalize(X, Y)
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1.,
                          lengthscale=1.)
    gp = GPy.models.WarpedGP(X, Y, kernel=kernel,
                             warping_terms=2)
    rogp = rogp.ROGP(gp, norm=norm)

    return rogp


class TestWarping:

    def test_warping_numeric(self, rogp):
        Y = np.random.uniform(size=(5, 1))
        Z_rogp = rogp.warp(Y)
        Z_gpy = rogp.gp.warping_function.f(Y)

        assert np.all(Z_rogp == Z_gpy)

        GRAD_rogp = rogp.warp_deriv(Y)
        GRAD_gpy = rogp.gp.warping_function.fgrad_y(Y)

        assert np.all(GRAD_rogp == GRAD_gpy)


class TestPrediction:

    def test_predict(self, rogp):
        X = np.random.uniform(size=(5, 1))
        X_norm = rogp.norm.X_norm.normalize(X)
        rogp.gp.predict_in_warped_space = False
        mu_rogp, cov_rogp = rogp.predict(X)
        mu_gpy, cov_gpy = rogp.gp.predict(X_norm)
        mu_gpy = rogp.norm.Y_norm.inverse_mean(mu_gpy)
        cov_gpy = rogp.norm.Y_norm.inverse_mean(cov_gpy)
        # assert np.all(mu_rogp == pytest.approx(mu_gpy))
        assert np.all(cov_rogp == pytest.approx(cov_gpy))
