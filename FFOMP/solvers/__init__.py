"""
The solvers for the fitting job
===============================

This package contains both a set of built-in solvers for the FFOMP model
parameter fitting tasks and some utilities for constructing new solvers. All
solvers are in fact functions that is able to accept a linear list of equations
as ``Eqn`` instances and a list of ``ModelParam`` instances to return a list
for the numerical values of the fitted parameters.

The default solvers are stored in this top-level module as a dictionary
``SOLVERS_DICT`` and the actual solvers are implemented and documented in
modules

.. autosummary::
    :toctree:

    linear
    scipyoptimize
    pyswarm

And their implementation has utilized the utilities in the module,

.. autosummary::
    :toctree:

    utils

"""
