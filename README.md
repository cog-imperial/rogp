# ROGP

ROGP is a tool for including the mean and covariance function of Gaussian
Processes in [Pyomo]() models. It was developed for the work outlined in the
following paper:
[A robust approach to warped Gaussian process-constrained
optimization](https://arxiv.org/abs/2006.08222)

# Usage

    x, y = generate_data(N, noise, dist=dist)
    X, Y, norm = normalize(x, y)

    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    gp = GPy.models.GPRegression(X, Y, kernel)
    gp.optimize(messages=True)

    gp = rogp.from_gpy(gp, norm=norm)
    

    m = pe.ConcreteModel()
    m.x = pe.Var(range(n))
    gp.predict_mu(x)
    gp.predict_cov(x)


# Acknowledgements
    This work was funded by the Engineering \& Physical Sciences Research
    Council (EPSRC) Center for Doctoral Training in High Performance Embedded
    and Distributed Systems (EP/L016796/1) and an EPSRC/Schlumberger CASE
    studentship (EP/R511961/1, voucher 17000145).
