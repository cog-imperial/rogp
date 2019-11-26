#!/usr/bin/env
import pytest
import numpy as np


@pytest.fixture

def gp(scope='module'):
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
    gp = GPy.models.GPRegression(X, Y, kernel=kernel)
    rogp = rogp.Standard(gp, norm=norm)

    return rogp


def wgp(scope='module'):
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
    rogp = rogp.Standard(gp, norm=norm)

    return rogp


class TestWarping:

    def test_warping_numeric(self, gp):
        Y = np.random.uniform(size=(5, 1))
        Z_rogp = wgp.warp(Y)
        Z_gpy = wgp.gp.warping_function.f(Y)

        assert np.all(Z_rogp == Z_gpy)

        GRAD_rogp = wgp.warp_deriv(Y)
        GRAD_gpy = wgp.gp.warping_function.fgrad_y(Y)

        assert np.all(GRAD_rogp == GRAD_gpy)


class TestPrediction:

    def test_predict(self, wgp):
        X = np.random.uniform(size=(5, 1))
        x = wgp.norm.x.normalize(X)
        wgp.gp.predict_in_warped_space = False
        mu_rogp, cov_rogp = wgp.predict(X)
        mu_gpy, cov_gpy = wgp.gp.predict(x)
        mu_gpy = wgp.norm.y.inverse_mean(mu_gpy)
        cov_gpy = wgp.norm.y.inverse_mean(cov_gpy)
        # assert np.all(mu_rogp == pytest.approx(mu_gpy))
        assert np.all(cov_rogp == pytest.approx(cov_gpy))

    def test_predict_numeric(self, gp):
        x = np.random.uniform(size=(5, 1), low=-1., high=1.)
        X = gp.norm.x.inverse_mean(x)
        mu_rogp, cov_rogp = gp.predict(X)
        mu_rogp = mu_rogp.astype(float)
        cov_rogp = cov_rogp.astype(float)
        mu_gpy, cov_gpy = gp.gp.predict(x, full_cov=True)
        mu_gpy = gp.norm.y.inverse_mean(mu_gpy)
        cov_gpy = gp.norm.y.inverse_variance(cov_gpy)
        print(cov_rogp, cov_gpy)
        assert np.all(mu_rogp == pytest.approx(mu_gpy))
        assert np.all(cov_rogp == pytest.approx(cov_gpy))


