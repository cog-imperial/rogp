#!/usr/bin/env
import GPy
import plotly.offline as pltoff


def plot(gp):
    GPy.plotting.change_plotting_library('plotly_offline')
    fig = gp.plot()[0]
    pltoff.plot(fig)

def plot_warping(gp):
    GPy.plotting.change_plotting_library('plotly_offline')
    fig = gp.warping_function.plot()[0]
    pltoff.plot(fig)
