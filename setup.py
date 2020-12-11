#!/usr/bin/env
from setuptools import setup, find_packages

setup(
    name='ROGP',
    version='0.1dev',
    author='Johannes Wiebe',
    author_email='j.wiebe17@imperial.ac.uk',
    packages=['rogp'],
    install_requires=['GPy','pyomo','numpy','scipy','pandas', 'matplotlib']
)
