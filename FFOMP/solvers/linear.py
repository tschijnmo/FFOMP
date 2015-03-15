"""
Linear least-square solvers
===========================

This module contains specialized solvers that works for the cases where the
value of the modelled properties depend linearly on all the model parameters.
This is definitely the most robust solvers among all the solvers, just it
requires the model to be linear. And all the solvers in this modules does not
respect the initial guess for the parameters for sure, since all of them are
based on singular value decomposition of the matrices and not iterative.

Two solvers are provided by this module,

.. autosummary::
    :toctree:

    numpy_lstsq
    r_lm

and they are based on the utility functions in the module
:py:mod:`FFOMP.solver.utils`.

"""

import time

import numpy as np
from numpy.linalg import lstsq

from .utils import get_linear_sys, decompose_mat2cols, get_total_weight


#
# The plain least square solver based on numpy
# --------------------------------------------
#


def numpy_lstsq(**kwargs):
    """Generates a plain least-square solver based on numpy lstsq function

    This function will return a plain linear least-square solver. Note that
    this solver will not respect the given weights for the properties. And it
    works only for linear models for sure.

    All the keyword arguments to this function will be passed to the numpy
    ``numpy.linalg.lstsq`` function.

    :returns: The solver that can be called with the list of equations and
        parameters.
    :rtype: function
    """

    def solver(eqns, params):
        """The actual plain least square solver"""

        # First generates the linear system.
        mat, vec = get_linear_sys(eqns, params)

        print(
            'Invoking the numpy.linalg.lstsq function...'
            )
        start_time = time.process_time()

        res = lstsq(mat, vec, **kwargs)

        print(
            'Finished: {!s}sec.'.format(time.process_time() - start_time)
            )

        return res

    return solver


#
# Solvers bases on GNU R
# ----------------------
#


def r_lm(prop_vec_name='props', **kwargs):
    """Generates the linear solver based on RPy2

    This function will generate a solver that invokes the linear model fitting
    facility of GNU R based on the RPy2 interface. The weights will be
    respected for this sophisticated solver.

    All keyword arguments not for this function will be passed to the core R
    ``lm`` function.

    :param str prop_vec_name: The name for the property vector, default to
        ``props``, to be used in the left-hand side of the R formula.
    :returns: The linear solver based on R
    :rtype: function
    """

    # Import here so that users do not have to install R if they are not going
    # to use it.
    try:
        from rpy2.robjects import r, Formula
        from rpy2.robjects.packages import importr
    except ImportError:
        raise ImportError(
            'GNU R and RPy2 have to be installed to use the R solver!'
            )

    def solver(eqns, params):
        """The actual R solver"""

        # Generate the linear system first.
        mat, vec = get_linear_sys(eqns, params)

        # Decompose the matrix.
        coeff_vecs = decompose_mat2cols(mat, params)

        # Test the validity of the property vector name.
        if prop_vec_name in coeff_vecs:
            raise ValueError(
                'Invalid property vector name {}, conflicts with parameter '
                'name! '.format(prop_vec_name)
                )

        # Generate the R formula.
        fmla = Formula(''.join([
            prop_vec_name, ' ~ ',
            ' + '.join(coeff_vecs.keys()),
            ' - 1'
            ]))

        # Add the data vectors.
        env = fmla.environment()
        env[prop_vec_name] = vec
        for param_name, coeff_vec in coeff_vecs.items():
            env[param_name] = coeff_vec
            continue

        # Generates the weights vector.
        tot_weight = get_total_weight(eqns)
        weights_vec = np.array(
            (i.weight / tot_weight for i in eqns),
            dtype=np.float
            )

        print('Invoking the R lm function...\n\n')
        start_time = time.process_time()

        # Invoke the R solver.
        stats = importr('stats')
        fit = stats.lm(fmla, weights=weights_vec, **kwargs)

        print(
            'Finished: {!s}sec.\n'.format(time.process_time() - start_time)
            )

        # Print the summary.
        print('R modelling summary: \n')
        print(r.summary(fit))
        print('\n')

        # Return the values for the parameters.
        return None  # TODO: set to the actual expression.
