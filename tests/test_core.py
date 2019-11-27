#!/usr/bin/env
import pytest
import rogp
import numpy as np
import pyomo.environ as p
from rogp.util.numpy import _pyomo_to_np, _eval_p


@pytest.fixture
def gp(scope='module'):
    import json
    import GPy

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

    return rogp.from_gpy(gp, norm=norm)


@pytest.fixture
def wgp(scope='module'):
    import json
    import GPy

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

    return rogp.from_gpy(gp, norm=norm)


class TestWarping:

    def test_warping_numeric(self, wgp):
        Y = np.random.uniform(size=(5, 1))
        Z_rogp = wgp.warp(Y)
        Z_gpy = wgp.gp.warping_function.f(Y)

        assert np.all(Z_rogp == Z_gpy)

        GRAD_rogp = wgp.warp_deriv(Y)
        GRAD_gpy = wgp.gp.warping_function.fgrad_y(Y)

        assert np.all(GRAD_rogp == GRAD_gpy)


class TestPrediction:

    def test_predict_warped_observation(self, wgp):
        x_norm = np.random.uniform(size=(5, 1), low=-1., high=1.)
        x = wgp.norm.x.inverse_mean(x_norm)
        m = p.ConcreteModel()
        m.cons = p.ConstraintList()
        m.z = p.Var(range(len(x)))
        z = _pyomo_to_np(m.z)
        mu_rogp = wgp.predict_mu(x, z, m.cons)
        solver = p.SolverFactory('ipopt', solver_io='nl')
        res = solver.solve(m)
        wgp.gp.predict_in_warped_space = True
        mu_rogp = np.array(_eval_p(m.z))[:, None]
        mu_gpy, _ = wgp.gp.predict(x_norm, median=True)
        mu_gpy = wgp.norm.y.inverse_mean(mu_gpy)
        assert np.all(mu_rogp == pytest.approx(mu_gpy))

    def test_predict_warped_latent(self, wgp):
        x_norm = np.random.uniform(size=(5, 1), low=-1., high=1.)
        x = wgp.norm.x.inverse_mean(x_norm)
        mu_rogp, cov_rogp = wgp.predict_latent(x)
        mu_rogp = mu_rogp.astype(float)
        cov_rogp = cov_rogp.astype(float)
        cov_rogp = np.diag(cov_rogp)[:, None]
        wgp.gp.predict_in_warped_space = False
        mu_gpy, cov_gpy = wgp.gp.predict(x_norm)
        assert np.all(mu_rogp == pytest.approx(mu_gpy))
        assert np.all(cov_rogp == pytest.approx(cov_gpy))

    def test_predict_standard(self, gp):
        x_norm = np.random.uniform(size=(5, 1), low=-1., high=1.)
        x = gp.norm.x.inverse_mean(x_norm)
        mu_rogp, cov_rogp = gp.predict(x)
        mu_rogp = mu_rogp.astype(float)
        cov_rogp = cov_rogp.astype(float)
        mu_gpy, cov_gpy = gp.gp.predict(x_norm, full_cov=True)
        mu_gpy = gp.norm.y.inverse_mean(mu_gpy)
        cov_gpy = gp.norm.y.inverse_variance(cov_gpy)
        assert np.all(mu_rogp == pytest.approx(mu_gpy))
        assert np.all(cov_rogp == pytest.approx(cov_gpy))
