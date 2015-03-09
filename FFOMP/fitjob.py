"""
Fit jobs to perform
===================

This is the highest-level model of the package, which contains the definition
of fit jobs that we can perform to obtain the best parameters for models.

.. autosummary::
    :toctree:

    FitJob

"""


import sys
import itertools
import collections
from collections import abc

from sympy import Expr

from .solvers import solvers_dict
from ._iterutil import flatten_zip, map_nested


#
# Equation and model parameter classes
# ------------------------------------
#
# Due to their relative simplicity, they are implemented as named tuples rather
# than full classes.
#


Eqn = collections.namedtuple(
    'Eqn', [
        'ref_val',  # Numeric reference value
        'modelled_val',  # Symbolic modelled value
        'weight'  # The weight for the equation
        ])

ModelParam = collections.namedtuple(
    'ModelParam', [
        'symb',  # The sympy symbol used for the symbol.
        'upper',  # The upper limit for the value.
        'lower',  # The lower limit for the value.
        'init_guess',  # The initial guess.
        ])


#
# The main class definition
# -------------------------
#


class FitJob:

    """
    Jobs to fit the parameters in some models to a data set

    Instances of this class represents fit jobs that this package is able to
    perform and it is the primary class that users are going to interact with.
    The primary actions that need to be performed on its instances are

    1. Addition of the raw data
    2. Application of some models
    3. Initiating the fitting process by a supported solver
    4. Extracting of the results of interest.

    """

    __slot__ = [
        '_raw_data',  # The raw data points
        '_models',  # The models
        '_prop_comp',  # The comparison for properties
        '_eqns',  # Linear list of all equations
        '_model_params',  # Linear list of all model parameters
        '_fit_res',  # The result of the fitting
        ]

    #
    # The initializer
    # ^^^^^^^^^^^^^^^
    #

    def __init__(self):
        """Initializes a fit job

        This initializer will set the new fit job into an empty state, pending
        more actions to be performed on it.
        """

        # Here first the attributes of significance to indicating the current
        # state of the job are initialized to some fixed values since they are
        # important.
        self._raw_data = []
        self._models = None
        self._fit_res = None

        # Other properties, just to satisfy pylint.
        self._prop_comp = None
        self._eqns = None
        self._model_params = None

    #
    # Adding raw data points
    # ^^^^^^^^^^^^^^^^^^^^^^
    #

    def add_raw_data(self, raw_data_pnts):
        """Adds multiple raw data points to the fit job

        This method can be called to add raw data to the fit job. The raw data
        points should be dictionaries with property name as keys and property
        values as values.

        :param raw_data_pnts: An iterable for the raw data points for the
            fitting job.
        :returns: None
        :raises ValueError: if the the fitting job is no longer in raw data
            adding stage.
        """

        if self._models is not None:
            raise ValueError(
                'The fitting job is no longer in raw data adding stage!'
                )
        else:
            self._raw_data.extend(
                raw_data_pnts
                )

        return None

    #
    # Model application
    # ^^^^^^^^^^^^^^^^^
    #

    def apply_models(self, models, prop_merger=None):
        """Applies models to the raw data points

        This function has to be called after the raw data has been added. It
        will apply the given models to the raw data points that we have got and
        merge their result to give the modelled symbolic result, which will
        later be solved by adjusting the actual values of the symbols in the
        expressions to minimize their difference with the numeric reference
        values.

        :param models: An iterator of :py:class:`~.Model` subclass instances to
            be applied to the raw data points.
        :param prop_merger: A dictionary for the mergers for properties
            computed by multiple models. Its key should be the name of the
            property and the value should be functions that is going to be
            called by the two results computed by two models and should return
            the combined result for that property.
        :returns None:
        """

        prop_merger = prop_merger or {}

        # Job status checking.
        if len(self._raw_data) == 0:
            raise ValueError(
                'The raw data have not been added!'
                )
        if self._models is not None:
            raise ValueError(
                'The models has been applied for the job!'
                )

        # Add the iterable of models to the current job.
        self._models = list(models)

        # Apply the models.
        #
        # The comparison of the reference and modelled properties, as a
        # dictionary with key being the property name, and list of pairs of
        # reference and modelled values of the property for all data points as
        # values. It is organized in this way for the convenience of the
        # normalization of the weights for different properties.
        self._prop_comp = collections.defaultdict(list)

        # For each data point, apply the models one-by-one
        for data_pnt in self._raw_data:
            res = {}  # Model application result for the current point
            for model in self._models:

                # Add the result of this model to the final result for the
                # current data point one-by-one.
                for prop, val in model(data_pnt).items():
                    if prop not in res:
                        # New property.
                        res[prop] = val
                    else:
                        # An existing property.
                        try:
                            merger = prop_merger[prop]
                        except KeyError:
                            raise ValueError(
                                'Property {} has been computed by multiple '
                                'models without given a merger!'.format(prop)
                                )
                        res[prop] = merger(res[prop], val)
                    # Continue to the next property.
                    continue

                # Continue to the next model.
                continue

            # After all the models has been applied to this data point, add the
            # result to comparison.
            for prop, val in res:
                if prop in data_pnt:
                    # If the computed property is in the raw data.
                    self._prop_comp[prop].append(
                        (data_pnt[prop], val)
                        )
                else:
                    # When the computed property does not have a reference
                    # value, just go on to the next property.
                    continue

            # Continue to the next data point.
            continue

        return None

    #
    # Perform the actual fitting
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^
    #

    def perform_fit(self, solver='', weights=None, **kwargs):
        """Performs the actual fitting

        This function would invoke the requested solver to solve the problem of
        adjusting the values of the model parameters to get the best agreement
        with the reference value.

        :param solver: The solver that is going to be used for the regression.
            It can be given as a string to invoke built-in solvers in the
            package :py:mod:`~.solvers`. Callable values can also be given,
            which are going to be called directly with the linear list of all
            the equations and the linear list of model parameters, and should
            return a linear list for the values of the symbols.
        :param dict weights: The weights for the properties. It can be unset to
            indicate that the fit is not weighted. Or it needs to be a
            dictionary with all the property names to be matched as keys and
            the weight for that property, which need not be normalized.
        :param kwargs: Any other key word arguments will be passed to the
            actual solver. Different solvers will accept different options.
        :returns: None

        .. warning::

            Not all solvers will respect the given weights, upper or lower
            limits, or even the initial guess.
        """

        # Check if the job is in the correct state.
        if self._models is None:
            raise ValueError(
                'The fitting job has not had models applied. Not ready for '
                'fitting.'
                )
        assert self._prop_comp is not None

        # Now we need to linearize the property comparison and the set of all
        # symbols to linear lists of Eqn and ModelParam instances for easier
        # manipulation of the solvers.
        #
        # First, the property comparison.
        unnormalized_eqns = []
        # The comparison is linearized by listing all comparisons for all data
        # points for the properties in turn.
        for prop, comp in self._prop_comp.items():

            # First get the weight for the current property.
            try:
                raw_wgt = (
                    (1.0, ) if weights is None else (weights[prop], )
                    )
            except KeyError:
                raise ValueError(
                    'The weights for properties cannot be partially set.'
                    )

            # Then linearize the possibly tensorial data for the data points in
            # turn, to get the linear list of equation left and righ hand
            # sides.
            try:
                lin_eqn_sides = list(itertools.chain.from_iterable(
                    flatten_zip(i, j)
                    for i, j in comp
                    ))
            except ValueError:
                raise ValueError(
                    'Unable to match reference and model values '
                    'for property {}!'.format(prop)
                    )

            # Add the equations for the current property by using the weight
            # scaled with the number of equations. Note that this weight is
            # still unnormalized, just it is free from the impact of the
            # possibly different numbers of equations for different properties.
            scaled_wgt = raw_wgt / len(lin_eqn_sides)
            unnormalized_eqns.extend(
                Eqn(ref_val=ref, modelled_val=modelled, weight=scaled_wgt)
                for ref, modelled in lin_eqn_sides
                )

            # Now go on to the next propery.
            continue

        # After all the properties has been added. We need to normalize the
        # weights and assign the finalized equations to the correct attribute.
        total_weight = sum(i.weight for i in unnormalized_eqns)
        self._eqns = [
            i._replace(weight=(i.weight / total_weight))
            for i in unnormalized_eqns
            ]

        # Next we need to linearize all the symbols for the model parameters as
        # well.
        self._model_params = list(itertools.chain.from_iterable(
            i.model_params for i in self._models
            ))

        # Finally, after all the preparation, invoke the solver.
        #
        # First get the solver functions based on user input.
        if isinstance(solver, abc.Callable):
            # For user-defined solvers.
            solver_func = solver
        else:
            # For built-in solvers.
            try:
                solver_func = solvers_dict[solver]
            except KeyError:
                raise ValueError(
                    'Unsupported solver {}!'.format(solver)
                    )

        # Invoke the solver function.
        self._fit_res = solver_func(
            self._eqns, self._model_params, kwargs
            )

        return None

    #
    # Presentation of fitting result
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #

    def get_raw_params(self):
        """Gets the raw values of the parameters after the fitting

        This function can be used to get the values of the model parameters
        that was optimized during the fitting.

        :returns: The dictionary containing the symbols for the model
            parameters as keys and the numerical value of the optimized
            parameters as values.
        :rtype: dict
        """

        if self._fit_res is None:
            raise ValueError(
                'The fitting result is not yet available!'
                )
        else:
            return dict(zip(
                (i.symb for i in self._model_params),
                self._fit_res
                ))

    def present_res(self, output=None):
        """Present the results prettily

        The actual presentation is achieved by calling the ``present`` method
        of the models with the list for the result for the model parameters for
        the given model.

        :param output: The output stream, the standard output by default.
        """

        output = output or sys.stdout
        for model, res in zip(self._models, self.fit_res4models):
            model.present(res, output)
            continue

        return None

    @property
    def fit_res4models(self):
        """Gets the fit result for each model

        This property will return a list of lists of the fitted values of the
        model parameters for each model.
        """

        # Check the state of the instance.
        if self._fit_res is None:
            raise ValueError(
                'The fitting result is not yet available!'
                )

        ret_val = []

        n_presented = 0
        for model in self._models:
            n_symbs = len(model.model_params)
            ret_val.append(
                self._fit_res[n_presented:n_presented + n_symbs]
                )
            n_presented += n_symbs
            continue

    #
    # Comparison of fitted and original values
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #

    def compare_prop(self, ref_prop, prop):
        """Compares the reference property value and modelled property value

        This function is going to return a list of triples for the comparison
        of the reference value and the fitted value for the fitting job, with
        the reference property leading as the coordinate.

        :param ref_prop: The reference property to be used as the leading
            column of the comparison. It can be set to None to use just the
            index for the data points.
        :param props: The property to compare.
        :returns: A list of reference property value, reference property value
            and the modelled property value triples.
        :rtype: list
        """

        # Check the state of the job.
        if self._fit_res is None:
            raise ValueError(
                'The fitting result is not yet available!'
                )

        # Form the dictionary for the substitution matrix
        subs = dict(zip(
            self._model_params, self._fit_res
            ))

        def symb2num(symb):
            """Converting symbolic scalar value to numeric one

            Of cause based on the result of the fitting.
            """

            if isinstance(symb, Expr):
                return symb.evalf(subs=subs)
            else:
                return symb

        # Get the reference property
        if ref_prop is None:
            # The default coordinate.
            ref_prop_vals = list(range(0, len(self._raw_data)))
        else:
            # User-defined coordinate.
            try:
                ref_prop_vals = [
                    i[ref_prop] for i in self._raw_data
                    ]
            except KeyError:
                raise ValueError(
                    'Invalid reference property {}'.format(ref_prop)
                    )

        # Get the comparison.
        try:
            prop_comp = self._prop_comp[prop]
        except KeyError:
            raise ValueError(
                'Invalid property to compare {}!'.format(prop)
                )

        # Return
        return [
            (i, j[0], map_nested(symb2num, j[1]))
            for i, j in zip(ref_prop_vals, prop_comp)
            ]
