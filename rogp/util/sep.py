#!/usr/bin/env
import utils
import bisect
import time
import numpy as np


class Projection():
    def __init__(self, xp, invcov):
        self.invcov = invcov
        self.xp = xp
        c = self.phi(0)
        a = (self.phi(1) + self.phi(-1))/2 - c
        b = self.phi(1) - a - c
        t = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        self.x = xp - np.ones(xp.shape)*t

    def phi(self, t):
        tv = np.ones(self.xp.shape)*t
        xp = self.xp
        invcov = self.invcov
        diff = xp - tv
        return np.matmul(np.matmul(diff.T, invcov), diff)[0, 0] - 1


class Node:
    def __init__(self, x, c, cov, warping, invcov=None):
        self.x = x
        self.N = x.shape[0]
        self.c = c
        self.cov = cov
        self.warping = warping
        if invcov is None:
            invcov = np.linalg.inv(cov)
        self.invcov = invcov
        self.ub = self.f(x)

    def f(self, x):
        hinv = self.warping.f_inv(x)
        return np.matmul(self.c, hinv)

    def project(self):
        projection = Projection(self.x, self.invcov)
        self.projection = projection.x
        self.lb = self.f(self.projection)
        return self.projection, self.lb

    def children(self):
        children = []
        for i in range(self.N):
            x = self.x.copy()
            x[i, 0] = self.projection[i, 0]
            children.append(Node(x, self.c, self.cov,
                                 self.warping, invcov=self.invcov))
        self.children = children
        return children


class Tree():
    def __init__(self, node):
        self.nodes = [node]
        self.ubs = [node.ub]

    def pop(self):
        node = self.nodes.pop()
        _ = self.ubs.pop()
        return node

    def insort(self, node):
        i = bisect.bisect_right(self.ubs, node.ub)
        self.ubs.insert(i, node.ub)
        self.nodes.insert(i, node)
        # self.ubs.append(node.ub)
        # self.nodes.append(node)

    def is_empty(self):
        return not bool(self.nodes)

    def ub(self):
        return max(self.ubs)

    def prune(self, lb):
        pass


def boundary_box_algorithm(cov, gp, c, eps=10e-3, max_iter=10000):
    ts = time.time()
    lb = float('-inf')
    x = np.diag(cov)[:, None]
    node = Node(x, c, cov, gp)
    ub = node.ub
    unexplored = Tree(node)
    n_iter = 0
    n_add = 0
    n_dontadd = 0
    while unexplored:
        node = unexplored.pop()
        node.project()
        print(lb, ub, abs((ub-lb)/ub))  # , node.x, node.projection)
        n_iter += 1
        # Update lower bound
        if node.lb > lb:
            lb = node.lb
            unexplored.prune(lb)

        # Add children or fathom
        for child in node.children():
            if child.ub >= lb:
                unexplored.insort(child)
                n_add += 1
            else:
                n_dontadd += 1
        if unexplored.is_empty():
            break
        ub = unexplored.ub()

        # Stop when converged
        if abs((ub - lb)/ub) <= eps:
            break
        if n_iter > max_iter:
            break
    return lb, ub, node, n_iter, time.time() - ts


if __name__ == '__main__':
    N_data = 50
    noise = 0.03
    wt = 2
    dist = 'nonuniform'
    gp = utils.train_warped_gp(N_data, noise, warping_terms=wt, dist=dist)
    wf = gp.gp.warping_function
