"""
Utilities for building solvers
==============================

Solvers are called with the list of symbolic equations and the list of all
model parameters to be fitted. In this module, utilities are provided for
translating these symbolic data to numeric functions that is convenient to be
used for the actual numeric solvers. This module is the interface between the
symbolic part and the numeric part of this package.

.. autosummary::
    :toctree:

    get_linear_sys
    decompose_mat2cols
    get_diff_funcs
    get_mds_funcs
    get_total_weight
    get_init_guess_vec

"""


import collections
import time

import numpy as np

from sympy import Mul, Add, Symbol, Number, Matrix, lambdify, sqrt


#
# Linear models
# -------------
#
# The utilities in this section is applicable for linear models only.
#


def get_linear_sys(eqns, params):
    """Gets the linear system corresponding to the symbolic equations

    Note that this function only work for models where the left-hand side of
    the equations all contain only linear terms with respect to the given model
    parameters. For these linear cases, this function will return a matrix
    :math:`\\mathbf{A}` and a vector :math:`\\mathbf{v}` such that the given
    equations can be written as

    .. math::

        \\mathbf{A} \\mathbf{x} = \\mathbf{v}

    with :math:`\\mathbf{x}` being the column vector of the values of the model
    symbols. Normally the matrix will have more rows than columns for over-
    determined fitting.

    :param eqns: A sequence of ``Eqn`` objects for the equations of the
        fitting.
    :param params: A sequence of the ``ModelParam`` objects for the parameters
        to be fitted.
    :returns: The matrix :math:`\\mathbf{A}` and the vector
        :math:`\\mathbf{v}`.
    :rtype: tuple
    :raises ValueError: if the system of equations are not linear.
    """

    # We treat the equations one-by-one, write rows of the matrix and
    # the vector one-by-one.
    n_params = len(params)
    n_eqns = len(eqns)
    mat = np.zeros((n_eqns, n_params), dtype=np.float)
    vec = np.empty((n_eqns, ), dtype=np.float)

    # Extract the symbols for the parameters and assort the result into a
    # dictionary for fast loop up of the location of the symbols.
    symbs = {
        param.symb: idx
        for idx, param in enumerate(params)
        }

    print('\nForming the matrix and vectors for the linear model...')
    start_time = time.process_time()
    for idx, eqn in enumerate(eqns):

        # First get the vector to the reference value of the equation.
        vec[idx] = eqn.ref_val

        # Get the symbolic expression.
        expr = eqn.modelled_val.simplify().expand()
        # Get its terms.
        if isinstance(expr, Add):
            terms = expr.args
        else:
            terms = [expr, ]

        # Loop over the terms to get the coefficients ahead of the symbols.
        for term in terms:

            # Split the term into a symbol and a coefficient.
            symb, coeff = _get_symb_w_coeff(term)

            if symb is None:
                # When we are treating a pure number term, we can move it to
                # the left-hand side of the equation.
                vec[idx] -= coeff
            else:
                # When we are going a symbol, we need to locate the symbol.
                try:
                    col_idx = symbs[symb]
                except KeyError:
                    raise ValueError(
                        'Unrecognised symbol {!r}'.format(symb)
                        )
                else:
                    mat[idx, col_idx] += coeff

            # Go on to the next term.
            continue

        # Go on to the next equation.
        continue

    print(
        'Finished: {!s}sec.'.format(time.process_time() - start_time)
        )

    # Return the matrix and the vector.
    return mat, vec


def _get_symb_w_coeff(term):
    """Gets the symbol and coefficient of a term

    This function works only for terms which are either a number or a symbol or
    a product of them, and splits the term into the symbol part and the
    coefficient part. For terms with no symbol, the None value will be returned
    for it.

    :param Expr term: The sympy term to be analysed.
    :returns: The symbol and coefficient (Python float) for the term.
    :rtype: tuple
    :raises ValueError: if the expression is not of the correct type.
    """

    if isinstance(term, Mul):
        # For multiplication terms, we should have got a product of a
        # symbol with one number.

        coeff = 1.0
        symb = None
        for arg in term.args:
            if isinstance(arg, Symbol):
                # For symbols, we should only have one symbol.
                if symb is None:
                    symb = arg
                else:
                    raise ValueError(
                        'Expression {!r} is non-linear!'.format(term)
                        )
            else:
                coeff *= _to_float(arg)
            # Continue to next factor.
            continue

    elif isinstance(term, Symbol):
        # For single symbol terms.

        coeff = 1.0
        symb = term

    else:
        # We can have a try to see if it is a pure numeric term.

        coeff = _to_float(term)
        symb = None

    return symb, coeff


def _to_float(expr):
    """Converts a sympy expression to a Python float

    The given expression must be a sympy ``Number`` isinstance, or ValueError
    will be raised.
    """

    res = expr.evalf()
    if isinstance(res, Number):
        return float(res)
    else:
        raise ValueError(
            'Expression {!r} is not a number!'.format(expr)
            )


def decompose_mat2cols(mat, params):
    """Decomposes the matrix for linear systems into vectors for parameters

    Some solvers, like that in GNU R, do not accept the rectangular matrix and
    vectors for linear systems, but rather, the coefficient vectors for each of
    the parameters are needed. This function will decompose the matrix into an
    ordered dictionary of vectors with the keys being the name of the
    parameters and values being the columns of the matrix, ie the coefficient
    vectors of the parameters.

    :param Array mat: The matrix for the linear system.
    :param params: The sequence of parameters for the linear system.
    :returns: An ordered dictionary for the coefficient vectors of the
        parameters.
    :rtype: OrderedDict
    """

    vecs = collections.OrderedDict()

    # Loop over the columns of the matrix along with the parameters.
    for col, param in zip(mat.T, params):
        vecs[param.symb.name] = col
        continue

    # Return the result.
    return vecs


#
# For non-linear models
# ---------------------
#
# For non-linear models, here utilities are provided for converting the
# equation array into numeric call-back functions for computing the component-
# wise deviation or the mean difference squared, and their respective first and
# second-order analytic derivative functions.
#


_LAMBDIFY_MODULES = [{'ImmutableMatrix': np.array}, 'numpy']
# The modules argument for the lambdify function.


def get_diff_funcs(eqns, params):
    """Gets the numeric functions for computing the component-wise differences

    This utility function is helpful for solvers that can be given the
    component-wise differences of the modelled and reference values directly.

    Since for this kind of solvers usually the solvers will already
    automatically scale the differences according to the number of equations,
    the weights attached to the equations will be normalized so that the least
    weight will become unity and the square root of the weights will be
    multiplied to the differences.

    :param eqns: A sequence of equations.
    :param params: A sequences of parameters.
    :returns: The difference value call-back function and the Jacobian call-
        back function. They can be called with the vector of the values of the
        parameters and returns the component-wise difference or the Jacobian
        for the difference vector function. The Jacobian has the derivative of
        the same component across the rows.
    :rtype: tuple
    """

    # First we need to get the scale factor.
    min_weight = min(i.weight for i in eqns)

    # Get the list of symbols for the model parameters.
    symbs = [i.symb for i in params]

    print('Forming the closure for computing the difference and Jacobian...')
    start_time = time.process_time()

    # Get the symbolic expression of the difference vector.
    diff_expr = [
        (
            (eqn.modelled_val - eqn.ref_val) *
            sqrt(eqn.weight / min_weight)
        ).simplify()
        for eqn in eqns
        ]

    # Get the symbolic expression of the Jacobian.
    jacobian_expr = [
        [comp.diff(i).simplify() for i in symbs]
        for comp in diff_expr
        ]

    # Lambdify the difference and Jacobian to numpy functions.
    lambdified_diff, lambdified_jacobian = [
        lambdify((symbs, ), Matrix(i), modules=_LAMBDIFY_MODULES)
        for i in [diff_expr, jacobian_expr]
        ]

    print(
        'Finished: {!s}sec.'.format(time.process_time() - start_time)
        )

    # Decorate and return the call-back functions.
    return lambda vec: lambdified_diff(vec).flatten(), lambdified_jacobian


def get_mds_funcs(eqns, params):
    """Gets the numeric functions for mean difference squared of all equations

    Frequently we would like to use some kind of general scalar numeric
    optimizers for parameter fitting. This function will automatically generate
    the expression for the weighted mean difference squared for all the
    equations. And return the numeric functions for computing that value, the
    gradient, and the Hessian.

    :param eqns: The sequence of equations to be solved.
    :param params: The sequence of model parameters.
    :returns: The numeric call-back functions for the mean difference squared
        and its Jacobian and Hessian. When called with the vector of values of
        the model parameters, the value function will return a float for the
        MDS, the Jacobian function will return a vector for the gradient, and
        the Hessian function will return a matrix.
    :rtype: tuple
    """

    # First get the total weight to normalize the weights of the equations.
    tot_weight = get_total_weight(eqns)

    # Get the list of symbols for the model parameters.
    symbs = [i.symb for i in params]

    print('Forming the closure for computing the MDS and derivatives...')
    start_time = time.process_time()

    # Form the expression for the weighted mean difference squared.
    mds_expr = sum(
        (eqn.weight / tot_weight) * (eqn.modelled_val - eqn.ref_val) ** 2
        for eqn in eqns
        )

    # Form the expression for the gradient.
    grad_expr = [
        mds_expr.diff(i).simplify()
        for i in symbs
        ]

    # Form the expression for the Hessian.
    hess_expr = [
        [i.diff(j).simplify() for j in symbs]
        for i in grad_expr
        ]

    # Lambdify the expressions.
    #
    # First lambdify the scalar function.
    lambdified_mds = lambdify(
        (symbs, ), mds_expr
        )
    # Next comes the tensorial quantities.
    lambdified_grad, lambdified_hess = [
        lambdify((symbs, ), Matrix(i), modules=_LAMBDIFY_MODULES)
        for i in [grad_expr, hess_expr]
        ]

    print(
        'Finished: {!s}sec.'.format(time.process_time() - start_time)
        )

    # Return the decorated results.
    return (
        lambdified_mds,
        lambda vec: lambdified_grad(vec).flatten(),
        lambdified_hess
        )


#
# General utilities
# -----------------
#
# Here some general utility functions useful across a lot of situations are
# implemented.
#


def get_total_weight(eqns):
    """Gets the total weight of the equations

    This functions sums the total weight of the equations. Then the normalized
    weight for each of the equations would be their given weight over the
    return value of this function.

    :param eqns: An iterable of the equations.
    :returns: The total weight of the equations.
    :rtype: float
    """

    return sum(
        eqn.weight for eqn in eqns
        )


def get_init_guess_vec(params):
    """Gets the initial guess vector

    This function will extract the initial guess data from the parameters and
    return them as a numpy vector.

    :param params: An iterable for the parameters.
    :returns: The vector for the initial guess.
    :rtype: Array
    """

    return np.array(
        [i.init_guess for i in params],
        dtype=np.float
        )


def get_bounds(params):
    """Gets the bounds of the parameters

    A list of ``(min, max)`` pairs will be returned. And the None value for the
    unbound parameters will be kept.

    :param params: An iterable for the model parameters.
    :returns: The list of bounds for the parameters.
    :rtype: list
    """

    return [
        (i.lower, i.upper) for i in params
        ]
