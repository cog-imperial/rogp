# ROGP

ROGP is a tool for including the mean and covariance function of Gaussian
Processes in [Pyomo]() models. It was developed for the work outlined in the
following paper:
[A robust approach to warped Gaussian process-constrained
optimization](https://arxiv.org/abs/2006.08222)

If you use this tool, please cite our paper as:

    @misc{Wiebe2020robust,
          title={A robust approach to warped Gaussian process-constrained optimization}, 
          author={Johannes Wiebe and Inês Cecílio and Jonathan Dunlop and Ruth Misener},
          year={2020},
          eprint={2006.08222},
          archivePrefix={arXiv},
          primaryClass={math.OC}
    }

## Installation

    python -m pip install git+https://github.com/johwiebe/rogp.git

## Usage

    # Normalize training set x, y
    norm = rogp.Normalizer()
    norm.scale_by(x, y)
    X, Y = norm.normalize(x, y)

    # Train GP using GPy
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    gp = GPy.models.GPRegression(X, Y, kernel)
    gp.optimize(messages=True)

    # Create ROGP object
    gp = rogp.from_gpy(gp, norm=norm)

    # Create Pyomo model and variable
    m = pe.ConcreteModel()
    m.x = pe.Var(range(n))

    # Generate Pyomo expressions
    xvar = rogp.pyomo_to_np(m.x)
    mu = gp.predict_mu(xvar)
    cov = gp.predict_cov(xvar)


## Acknowledgements
This work was funded by the Engineering \& Physical Sciences Research
Council (EPSRC) Center for Doctoral Training in High Performance Embedded
and Distributed Systems (EP/L016796/1) and an EPSRC/Schlumberger CASE
studentship (EP/R511961/1, voucher 17000145).

Author: Johannes Wiebe
