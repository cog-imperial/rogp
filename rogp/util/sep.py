#!/usr/bin/env
import bisect
import time
import itertools
import scipy.optimize
import numpy as np
import pandas as pd
import pyomo.environ as p
from . import normalizer
from rogp.util.numpy import _pyomo_to_np


class Box:
    def __init__(self, x, c, mu, cov, warping, invcov=None, hinv=None,
                 parent=None):
        self.x = x
        self.c = c
        self.mu = mu
        self.cov = cov
        self.warping = warping
        self.warp_inv = warping.warp_inv
        self.warp_inv_scalar = warping.warp_inv_scalar
        if invcov is None:
            invcov = np.linalg.inv(cov)
        self.invcov = invcov
        if hinv is None:
            self.ub, hinv_u = self.f(x[:, 1:2])
            self.lb, hinv_l = self.f(x[:, 0:1])
            self.hinv = np.concatenate((hinv_l, hinv_u), axis=1)
        else:
            self.hinv = hinv
            bounds = np.matmul(c, hinv)
            self.lb = bounds[0]
            self.ub = bounds[1]
        if parent is None:
            self.max_corner = np.where(x != mu)
            self.min_corner = np.where(x == mu)
        else:
            self.max_corner = parent.max_corner
            self.min_corner = parent.min_corner
        self.parent = parent

    def f(self, x):
        hinv = self.warp_inv(x)
        return np.matmul(self.c, hinv), hinv

    def get_children(self, lb, eps=1e-3):
        children = []
        axis = np.argmax(self.x[:, 1] - self.x[:, 0])
        midpoint = (self.x[axis, 1] + self.x[axis, 0])/2
        bracket = tuple(self.hinv[axis, :])
        hinv_new = self.warp_inv_scalar(midpoint, bracket)
        for i in range(2):
            x = self.x.copy()
            x[axis, i] = midpoint
            hinv = self.hinv.copy()
            hinv[axis, i] = hinv_new
            if self.on_boundary(x):
                child = Box(x, self.c, self.mu, self.cov, self.warping,
                            invcov=self.invcov, hinv=hinv, parent=self)
                if child.ub >= lb:
                    children.append(child)
        return children

    def on_boundary(self, x=None):
        if x is None:
            x = self.x

        # diff = x[self.max_corner] - self.mu
        # c = np.matmul(np.matmul(diff.T, self.invcov), diff)[0, 0]
        # if c < 1:
        #     return False
        # diff = x[self.min_corner] - self.mu
        # c = np.matmul(np.matmul(diff.T, self.invcov), diff)[0, 0]
        # if c > 1:
        #     return False
        # return True

        cons_vals = []
        for corner in itertools.product(*zip(x[:, 0], x[:, 1])):
            diff = np.array(corner)[:, None] - self.mu
            c = np.matmul(np.matmul(diff.T, self.invcov), diff)[0, 0]
            cons_vals.append(c)
        return min(cons_vals) <= 1 and max(cons_vals) >= 1

        # for i in range(self.N):
        #     xm = self.mu[i, 0] - np.sqrt(np.diag(self.cov)[i])
        #     # if self.projection[i, 0] - ()self.mu[i, 0] > eps:
        #     if self.projection[i, 0] - xm > eps and self.grad[i, 0] > eps:
        #         x = self.x.copy()
        #         hinv = self.hinv.copy()
        #         x[i, 0] = self.projection[i, 0]
        #         hinv[i, 0] = self.projection_hinv[i, 0]
        #         child = Node(x, self.c, self.mu, self.cov,
        #                      self.warping,
        #                      invcov=self.invcov,
        #                      hinv=hinv, parent=self)
        #         if child.ub >= lb:
        #             children.append(child)
        # self.children = children


class Node:
    def __init__(self, x, c, mu, cov, warping, invcov=None, hinv=None,
                 parent=None):
        self.x = x
        self.N = x.shape[0]
        self.c = c
        self.mu = mu
        self.cov = cov
        self.warping = warping
        if invcov is None:
            invcov = np.linalg.inv(cov)
        self.invcov = invcov
        if hinv is None:
            self.ub, self.hinv = self.f(x)
        else:
            self.hinv = hinv
            self.ub = np.matmul(c, hinv)
        self.parent = parent

    def f(self, x):
        hinv = self.warping(x)
        return np.matmul(self.c, hinv), hinv

    def phi(self, x):
        diff = x - self.mu
        return np.matmul(np.matmul(diff.T, self.invcov), diff)[0, 0]

    def solve_quadratic(self, e):
        diff = self.x - self.mu
        c = np.matmul(np.matmul(diff.T, self.invcov), diff) - 1
        b = 2*np.matmul(np.matmul(diff.T, self.invcov), e)
        a = np.matmul(np.matmul(e.T, self.invcov), e)
        if (b**2 - 4*a*c) >= 0:
            return (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        else:
            return None

    def _project(self, e):
        t = self.solve_quadratic(e)
        if t is not None:
            projection = self.x + e*t
            grad = np.matmul(self.invcov, projection - self.mu)
            return projection, grad
        else:
            return None, None

    def project(self):
        eps = 1e-4
        diff = self.x - self.mu
        sig = 1.0
        t = None
        n_iter = 0
        # Make sure we're hitting the ellipsoid
        # TODO: Replace this by linesearch
        while t is None:
            e = -diff - np.sqrt(np.diag(self.cov)[:, None])*sig
            t = self.solve_quadratic(e)
            sig = sig/1.01
            n_iter += 1
            if n_iter > 10000:
                import ipdb; ipdb.set_trace()
        projection, grad = self._project(e)
        e_original = e.copy()
        n_iter = 0
        # Check if any element of gradient is negative
        if np.any(grad < -eps):
            def f(s, e, i):
                e[i] = s
                projection, grad = self._project(e)
                if grad is None:
                    return -1
                return np.min(grad)
            i = np.where(grad < 0)
            # Walk along axis for which gradient is negative until it becomes
            # zero
            if f(0, e, i) > 0:
                # res = scipy.optimize.root_scalar(f, (e, i), bracket=(-0.1, 0))
                try:
                    s_min = np.min(e_original[i])
                    res = scipy.optimize.root_scalar(f, (e, i),
                                                     bracket=(s_min, 0))
                except:
                    import ipdb; ipdb.set_trace()
                e[i] = res.root
                projection, grad = self._project(e)
            else:
                # Set e[i] = 0.5*e[i]
                # walk along on major axis (p.projection - c.x == 0) until
                # gradient = 0
                dd = self.parent.projection - self.x
                j = np.where(dd == 0)
                dd[i] = dd[i]/2
                # Fix s_min
                s_min = -0.5
                try:
                    res = scipy.optimize.root_scalar(f, (dd, j), bracket=(s_min,0))
                    dd[j] = res.root
                    projection, grad = self._project(dd)
                    e = dd
                except:
                    import ipdb; ipdb.set_trace()

        self.projection, self.grad = self._project(e)
        self.e = e
        self.lb, self.projection_hinv = self.f(self.projection)

        for k in range(self.x.shape[0]):
            if self.x[k] - self.projection[k] < eps and not grad[k] < eps:
                import ipdb; ipdb.set_trace()

        return self.projection, self.lb

    def get_children(self, lb, eps=1e-3):
        children = []
        for i in range(self.N):
            xm = self.mu[i, 0] - np.sqrt(np.diag(self.cov)[i])
            # if self.projection[i, 0] - ()self.mu[i, 0] > eps:
            if self.projection[i, 0] - xm > eps and self.grad[i, 0] > eps:
                x = self.x.copy()
                hinv = self.hinv.copy()
                x[i, 0] = self.projection[i, 0]
                hinv[i, 0] = self.projection_hinv[i, 0]
                child = Node(x, self.c, self.mu, self.cov,
                             self.warping,
                             invcov=self.invcov,
                             hinv=hinv, parent=self)
                if child.ub >= lb:
                    children.append(child)
        self.children = children
        return children


class Tree():
    def __init__(self, mu, cov, wf, c):
        self.lb = float('-inf')
        x = np.sqrt(np.diag(cov)[:, None]) + mu
        self.mu = mu
        root = Node(x, c, mu, cov, wf)
        self.nodes = [root]
        self.ubs = [root.ub]
        self.ub = root.ub
        self.x_lb = None
        self.x_ub = root.x

    def pop(self):
        node = self.nodes.pop()
        _ = self.ubs.pop()
        return node

    def insort(self, node):
        i = bisect.bisect_right(self.ubs, node.ub)
        self.ubs.insert(i, node.ub)
        self.nodes.insert(i, node)

    def is_empty(self):
        return not bool(self.nodes)

    def update_ub(self):
        self.ub = self.ubs[-1]
        self.x_ub = self.nodes[-1].x

    def prune(self):
        pass

    def solve(self, eps=1e-3, max_iter=10000):
        ts = time.time()
        n_iter = 0
        data = []
        data2 = []
        while self.nodes:
            # import ipdb; ipdb.set_trace()
            node = self.pop()
            # Project node onto ellipsoid
            node.project()
            n_iter += 1
            # Update lower bound
            if node.lb > self.lb:
                self.lb = node.lb
                self.x_lb = node.projection
                self.prune()

            # Add children or fathom
            for child in node.get_children(self.lb):
                # if child.ub > self.ub:
                #     import ipdb;ipdb.set_trace()
                self.insort(child)
            # Stop if no nodes left to explore
            if self.is_empty():
                break
            # Update upper bound
            self.update_ub()
            if n_iter % 100 == 0:
                print(self.lb, self.ub, abs((self.ub-self.lb)/self.ub),
                      node.x.flatten(), node.projection.flatten(),
                      node.grad.flatten())
            data.append(node.x.flatten())
            data2.append(node.projection.flatten())

            # Stop when converged
            if abs((self.ub - self.lb)/self.ub) <= eps:
                break
            # Or when maximum iterations are reached
            if n_iter > max_iter:
                break
                # import ipdb; ipdb.set_trace()
        self.n_iter = n_iter
        self.time = time.time() - ts
        df = pd.DataFrame(np.array(data))
        df['type'] = 'corner'
        df2 = pd.DataFrame(np.array(data2))
        df2['type'] = 'projection'
        dfmu = pd.DataFrame(self.mu.T)
        dfmu['type'] = 'mean'
        dfl = df.append(df2)
        dfl = dfl.append(dfmu)

        return self.lb, self.ub, node, n_iter, self.time, dfl


class BoxTree():
    def __init__(self, mu, cov, wf, c):
        self.mu = mu
        self.lb = float('-inf')
        rad = np.sqrt(np.diag(cov)[:, None])
        xmin = mu - rad
        xmax = mu + rad
        self.nodes = []
        self.ubs = []
        lb = float('-inf')
        for corner in itertools.product(*zip(xmin, xmax)):
            cmin = np.minimum(np.array(corner), mu)
            cmax = np.maximum(np.array(corner), mu)
            x = np.concatenate((cmin, cmax), axis=1)
            box = Box(x, c, mu, cov, wf)
            lb = max(lb, box.lb)
            self.insort(box)
        self.ub = self.ubs[-1]
        self.lb = lb

    def pop(self):
        node = self.nodes.pop()
        _ = self.ubs.pop()
        return node

    def insort(self, node):
        i = bisect.bisect_right(self.ubs, node.ub)
        self.ubs.insert(i, node.ub)
        self.nodes.insert(i, node)

    def is_empty(self):
        return not bool(self.nodes)

    def update_ub(self):
        self.ub = self.ubs[-1]
        self.x_ub = self.nodes[-1].x

    def prune(self):
        pass

    def solve(self, eps=1e-3, max_iter=10000):
        ts = time.time()
        n_iter = 0
        while self.nodes:
            # import ipdb; ipdb.set_trace()
            node = self.pop()
            # Project node onto ellipsoid
            n_iter += 1
            # Update lower bound
            if node.lb > self.lb:
                self.lb = node.lb
                self.x_lb = node.projection
                self.prune()

            # Add children or fathom
            for child in node.get_children(self.lb):
                # if child.ub > self.ub:
                #     import ipdb;ipdb.set_trace()
                self.insort(child)
                if child.lb > self.lb:
                    self.lb = child.lb
            # import ipdb; ipdb.set_trace()
            # print(self.nodes[-1].x)
            # Stop if no nodes left to explore
            if self.is_empty():
                break
            # Update upper bound
            self.update_ub()
            if n_iter % 100 == 0:
                print(n_iter, self.lb, self.ub, abs((self.ub-self.lb)/self.ub))
                # node.x.flatten())

            # Stop when converged
            # if abs((self.ub - self.lb)/self.ub) <= eps:
            if abs((self.ub - self.lb)/self.ub) <= eps:
                break
            # Or when maximum iterations are reached
            if n_iter > max_iter:
                break
                # import ipdb; ipdb.set_trace()
        self.n_iter = n_iter
        self.time = time.time() - ts

        return self.lb, self.ub, node, n_iter, self.time


if __name__ == '__main__':
    N_data = 50
    noise = 0.03
    wt = 2
    dist = 'nonuniform'
    df = pd.DataFrame.from_csv('gp_data.csv')
    gp = utils.train_warped_gp(N_data, noise, warping_terms=wt, dist=dist)
    wf = gp.gp.warping_function
