"""
Solvers based on functions in the scipy.optimize module
=======================================================

This module contains solvers that is based on the functions in the
``scipy.optimize`` module.

.. autosummary::
    :toctree:

    so_minimize
    so_leastsq

"""


import time

from scipy.optimize import minimize, leastsq

from .utils import (
    get_diff_funcs, get_mds_funcs, get_init_guess_vec, get_bounds
    )


#
# The solver based on the general minimization function
# -----------------------------------------------------
#


def so_minimize(use_grad=True, use_hess=True, use_bounds=True, **kwargs):
    """Generates the solver based on the minimize function

    This function will generate a solver based on the general
    ``scipy.optimize.minimize`` function for minimizing scalar object
    functions. All unrecognised keyword arguments will be given to that
    function. Note that ``jac``, ``hess``, and ``bounds`` arguments will be
    automatically generated and no need to be specified.

    :param bool use_grad: If analytic gradient is to be used.
    :param bool use_hess: If analytic hessian is to be used.
    :param bool use_bounds: If bounds is to be used.
    :returns: A solver based on the minimize function.
    """

    def solver(eqns, params):
        """The actual solver"""

        # Generate the functions
        val_func, grad_func, hess_func = get_mds_funcs(eqns, params)

        # Generate the initial guess.
        init_guess = get_init_guess_vec(params)

        print('Invoking the scipy.optimize.minimize function...\n\n')
        start_time = time.process_time()

        res = minimize(
            val_func, init_guess,
            jac=grad_func if use_grad else False,
            hess=hess_func if use_hess else None,
            bounds=get_bounds(params) if use_bounds else None,
            **kwargs
            )

        print(
            '\n\nFinished: {!s}sec.'.format(time.process_time() - start_time)
            )

        # Return the result.
        if not res.success:
            print(
                'WARNING: THE SOLVER HAS FAILED TO CONVERGE!'
                )

        return res.x


#
# The solver based on the linear square function
# ----------------------------------------------
#


def so_leastsq(use_jac=True, **kwargs):
    """Generates a solver based on the least square function

    A solver based on the ``scipy.optimize.leastsq`` function is to be
    generated. All unrecognised keyword arguments will be passed to that
    function. Just the Jacobian argument will be automatically generated based
    on the parameter.

    Note that this solver will not respect the bounds given for the parameters.

    :param bool use_jac: If the analytic Jacobian is to be used.
    :returns: A solver based on the nonlinear least square function.
    :rtype: function
    """

    def solver(eqns, params):
        """The actual solver"""

        # Generate the numeric call-backs.
        diff_func, diff_jac_func = get_diff_funcs(eqns, params)

        # Generate the initial guess.
        init_guess = get_init_guess_vec(params)

        print('Invoking the scipy.optimize.leastsq function...\n\n')
        start_time = time.process_time()

        res = leastsq(
            diff_func, init_guess,
            Dfun=diff_jac_func if use_jac else None,
            **kwargs
            )

        print(
            '\n\nFinished: {!s}sec.'.format(time.process_time() - start_time)
            )

        # return the result.
        if res[4] not in [1, 2, 3, 4]:
            print(
                'WARNING: THE SOLVER HAS NOT CONVERGED!'
                )

        return res[0]
