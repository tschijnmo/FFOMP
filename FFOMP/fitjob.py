"""
Fit jobs to perform
===================

This is the highest-level model of the package, which contains the definition
of fit jobs that we can perform to obtain the best parameters for models, along
with some utility functions.

.. autosummary::
    :toctree:

    FitJob
    read_data_from_yaml_glob
    add_elementwise

And they are based on the internal functions,

.. autosummary::
    :toctree:

    _linearize_comps2eqns
    _get_prop
    _compute_rms_deviation

"""


import sys
import itertools
import collections
from collections import abc
import math
import glob
import operator

from sympy import Expr

from yaml import load, YAMLError
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from ._iterutil import flatten_zip, map_nested, map_binary_nested


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
        '_fitted_data',
        # The list in bijective correspondence with _raw_data but contains the
        # properties that has been used for the fitting.
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
        self._fitted_data = None

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
        # dictionary with key being the property name, and list of triples of
        # reference and modelled values of the property, and the indices of the
        # data point in the raw data list, for all data points as values. It is
        # organized in this way for the convenience of the normalization of the
        # weights for different properties.
        self._prop_comp = collections.defaultdict(list)

        # For each data point, apply the models one-by-one
        for idx, data_pnt in enumerate(self._raw_data):
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
            for prop, val in res.items():
                if prop in data_pnt:
                    # If the computed property is in the raw data.
                    self._prop_comp[prop].append(
                        (data_pnt[prop], val, idx)
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

    def perform_fit(self, solver, weights=None):
        """Performs the actual fitting

        This function would invoke the requested solver to solve the problem of
        adjusting the values of the model parameters to get the best agreement
        with the reference value.

        :param solver: The solver that is going to be used for the regression.
            It needs to be callable values, which are going to be called
            directly with the linear list of all the equations and the linear
            list of model parameters, and should return a linear list for the
            values of the symbols. Built-in solvers, which solves most of the
            problems, exist in the package :py:mod:`~.solvers`.
        :param dict weights: The weights for the properties. It can be unset to
            indicate that the fit is not weighted. Or it needs to be a
            dictionary with all the property names to be matched as keys and
            the weight for that property, which need not be normalized.
        :returns: None

        .. warning::

            Not all solvers will respect the given weights, upper or lower
            limits, or even the initial guess. Need to refer to the
            documentation of the specific solver that is used.
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
        # First, linearize the property comparisons to equations.
        self._eqns = _linearize_comps2eqns(self._prop_comp, weights)

        # Next we need to linearize all the symbols for the model parameters as
        # well.
        self._model_params = list(itertools.chain.from_iterable(
            i.model_params for i in self._models
            ))

        # Finally, after all the preparation, invoke the solver.
        self._fit_res = solver(
            self._eqns, self._model_params
            )

        # Generate the values of the properties based on the fitted values of
        # the model parameters.
        #
        # Form the dictionary for the substitution matrix
        subs = dict(zip(
            (i.symb for i in self._model_params),
            self._fit_res
            ))
        # Perform actual substitution on the properties.
        fitted_props = {
            prop: [(
                map_nested(lambda x: float(
                    x.evalf(subs=subs) if isinstance(x, Expr) else x
                    ), i[1]), i[2]) for i in comp]
            for prop, comp in self._prop_comp.items()
            }
        # Construct the fitted properties for each of the raw data points.
        self._fitted_data = [{} for _ in self._raw_data]
        for prop, fitted_val in fitted_props.items():
            for val, idx in fitted_val:
                self._fitted_data[idx][prop] = val
                # Continue to the next data point.
                continue
            # Continue to the next property
            continue

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
        print('\nFitting results:\n')
        for model, res in zip(self._models, self.fit_res4models):
            model.present(res, output)
            continue
        print('\n\n')

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

        return ret_val

    #
    # Comparison of fitted and original values
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #

    def compare_prop(self, coord_prop, prop):
        """Compares the reference property value and modelled property value

        This function is going to return a triple for the comparison of the
        reference value and the fitted value for the fitting job, with the
        given coordinate property leading as the first field.

        For both the coordinate property and the property to compare, if only a
        string is given, it is going to be used for retrieving the property. It
        could also be a pair of a string and a callable, then the result of
        applying the callable to the raw property under the string name will be
        used as the value.

        :param coord_prop: The property to be used as the leading column of the
            comparison. It can be set to None to use just the index for the
            data points.
        :param prop: The property to compare.
        :returns: A triple of coordinate property value list, reference value
             list of the property to compare and the modelled value list.
        :rtype: tuple
        """

        # Check the state of the job.
        if self._fit_res is None:
            raise ValueError(
                'The fitting result is not yet available!'
                )

        # Allocate the lists.
        coord_prop_vals = []  # The values of the coordinate property.
        # The reference and modelled values of the compared property.
        ref_vals = []
        modelled_vals = []

        for idx in range(0, len(self._raw_data)):

            # First attempt to get the modelled value, if it is absent, it
            # means that this data point does not have got it. we can skip to
            # the next data point.
            try:
                modelled_vals.append(
                    _get_prop(self._fitted_data[idx], prop)
                    )
            except KeyError:
                continue
            else:
                # When we get here we know we are at a data point with the
                # property to be compared. So we can go ahead to retrieve the
                # reference value of the property and the reference property.

                # Get the reference property
                if coord_prop is None:
                    # The default coordinate.
                    coord_prop_vals.append(idx)
                else:
                    # User-defined coordinate.
                    coord_prop_vals.append(
                        _get_prop(self._raw_data[idx], coord_prop)
                        )

                # Get the reference value.
                ref_vals.append(
                    _get_prop(self._raw_data[idx], prop)
                    )

        # Return
        return (coord_prop_vals, ref_vals, modelled_vals)

    def print_prop_comps(self, coord_prop, props, fmt='{:^26}'):
        """Prints the modelled and reference values comparisons

        This method is built upon the :py:meth:`compare_prop` and is able to
        print the comparison of multiple properties as columns. The RMS
        deviation of the properties are also printed.

        :param coord_prop: The coordinate property.
        :param props: The sequence of properties to compare the reference and
            modelled values. The property values need to be scalar and all the
            properties needs to be applicable to the same set of data points.
        :param fmt: The format string for the data fields, it is advised that
            only field width is set.
        :returns: None
        """

        # The columns, with reference property leading.
        cols = []
        rms_devs = []  # The RMS deviations.

        # Add the columns one-by-one.
        for prop in props:
            # Get the comparison.
            comp = self.compare_prop(coord_prop, prop)
            # Test the reference property.
            if len(cols) == 0:
                cols.append(comp[0])
            else:
                if len(comp[0]) != len(cols[0]):
                    raise ValueError(
                        'Property {} is applicable to different data points '
                        'than other properties!'.format(prop)
                        )
            # Add the comparison.
            cols.extend(comp[1:])
            # Add the RMS deviation.
            rms_devs.append(
                _compute_rms_deviation(comp[1], comp[2])
                )
            # Continue to the next property.
            continue

        # Transpose the columns
        rows = sorted(zip(*cols), key=operator.itemgetter(0))

        # Printing.
        #
        # Get the complete format.
        compl_fmt = ' '.join(
            itertools.repeat(fmt, len(cols))
            )
        # Print header.
        prop_names = [
            i if isinstance(i, str) else i[0]
            for i in itertools.chain([coord_prop, ], props)
            ]
        titles = [prop_names[0], ] + list(itertools.chain.from_iterable(
            [i + ' (Reference)', i + ' (Modelled)'] for i in prop_names[1:]
            ))
        print('\n')
        print(compl_fmt.format(*titles))
        print(''.join(itertools.repeat('=', 80)))
        # Print the data points.
        for fields in rows:
            print(compl_fmt.format(*fields))
            continue
        print(''.join(itertools.repeat('=', 80)))
        # Print the RMS deviations.
        for title, rms_dev in zip(titles[1:], rms_devs):
            print(
                ' RMS deviation of {}: {}'.format(title, rms_dev)
                )
        print('\n\n')

        return None

    #
    # The drivers
    # ^^^^^^^^^^^
    #

    def fitting_driver(self, raw_data_pnts, models, solver,
                       weights=None, prop_merger=None):
        """Performs the fitting from the beginning to the fitting

        This is the driver function that incorporates the whole process up to
        the finish of the fitting process. This function definitely lacks the
        flexibility of the object-oriented interface. This function can be
        called at an empty fit job object. Then only post-processing tasks like
        extracting the results are needed to be performed on the job object.

        :param raw_data_pnts: The sequence of raw data points.
        :param models: The sequence of models to be applied.
        :param solver: The solver.
        :param solver_args: The arguments to the solver generator.
        :param weights: The weights for the properties.
        :param prop_merger: The mergers for the properties.
        :returns: The raw result for the
        """

        # pylint: disable=too-many-arguments

        # First add the data points.
        self.add_raw_data(raw_data_pnts)

        # Next apply the modes.
        self.apply_models(models, prop_merger=prop_merger)

        # Next perform the actual fitting.
        self.perform_fit(solver, weights=weights)

        # Return the raw results.
        return self.get_raw_params()

    def post_fitting_driver(self, prop_comps):
        """Performs the post-fitting jobs of outputting fitting results

        First all the models will be asked to present their fitted results.
        Then the requested property comparisons will be dumped.

        :param prop_comps: An iterable of property comparison specification,
            with each item being a pair of the coordinate property and a
            sequence of properties to compare. Optionally, a third field can be
            added which will be printed before the comparisons as a banner.
        :returns: None
        """

        # First present the results for the models.
        self.present_res()

        # Next dump the property comparisons.
        for i in prop_comps:
            print('\n')
            if len(i) > 2:
                print(i[2])
            self.print_prop_comps(i[0], i[1])
            continue

        return None

    def driver(self, raw_data_pnts, models, solver,
               weights=None, prop_merger=None, prop_comps=()):
        """The overall driver for the fitting from the beginning to the end

        This method is just a shallow wrapper over the
        :py:meth:`fitting_driver` and :py:meth:`post_fitting_driver` methods.
        """

        self.fitting_driver(
            raw_data_pnts, models, solver,
            weights=weights, prop_merger=prop_merger
            )

        self.post_fitting_driver(prop_comps)

        return None


#
# Utility functions
# -----------------
#


def read_data_from_yaml_glob(patt):
    """Reads data from the files described by the given glob

    This function will first get all the files matching the given glob and
    attempt to parse them with PyYAML. The PyYAML parsing output will be put
    into a list for the data points.

    :param str patt: The glob pattern for the file names.
    :returns: The file contents parsed by PyYAML.
    :rtype: list
    """

    data_pnts = []

    for f_name in glob.glob(patt):

        try:
            with open(f_name, 'r') as inp:
                content = inp.read()
        except IOError:
            raise IOError(
                'Corrupt input file {}!'.format(f_name)
                )

        try:
            data_pnts.append(
                load(content, Loader=Loader)
                )
        except YAMLError as exc:
            raise ValueError(
                'Invalid YAML file {}:\n{}'.format(f_name, exc)
                )

        # Continue to the next file
        continue

    return data_pnts


def add_elementwise(prop1, prop2):
    """Merges two nested sequences by adding the entries elementwise

    This function would be helpful to act as the property merger for properties
    that could be merged by adding the two values elementwise.
    """

    return map_binary_nested(
        operator.add, prop1, prop2
        )


#
# Internal functions
# ------------------
#


def _linearize_comps2eqns(comps, weights):
    """Linearizes property comparisons to equations

    This function will linearize the property comparisons to a linear list of
    equations, and properly scale the weights for the properties.

    :param dict comps: The modelled and reference property comparison
        dictionary, keys being the name of the property and values being the
        list of comparisons.
    :param weights: The weights of the properties. Can be set to None for
        unweighted fitting.
    :returns: The list of equations from flattening the all the comparisons.
        And the weights are all properly scaled. The weights are not normalized
        but are scaled so that all equations from the properties together holds
        the weights given by the user.
    :rtype: list
    """

    eqns = []

    # The comparison is linearized by listing all comparisons for all data
    # points for the properties in turn.
    for prop, comp in comps.items():

        # First get the weight for the current property.
        try:
            raw_wgt = (
                1.0 if weights is None else weights[prop]
                )
        except KeyError as exc:
            raise ValueError(
                'The weights for properties cannot be partially set. \n'
                'The weight for {} is missing!'.format(exc.args[0])
                )

        if raw_wgt == 0:
            # For weights given as integral value of zero, skip everything.
            continue

        # Then linearize the possibly tensorial data for the data points in
        # turn, to get the linear list of equation left and right hand
        # sides.
        try:
            lin_eqn_sides = list(itertools.chain.from_iterable(
                flatten_zip(i, j)
                for i, j, _ in comp
                ))
        except ValueError:
            raise ValueError(
                'Unable to match reference and model values '
                'for property {}!'.format(prop)
                )

        # Add the equations for the current property by using the weight scaled
        # with the number of equations. Note that this weight is still
        # unnormalized, just it is free from the impact of the possibly
        # different numbers of equations for different properties. Different
        # solvers are required to normalize the weights in their own preferred
        # way.
        scaled_wgt = raw_wgt / len(lin_eqn_sides)
        eqns.extend(
            Eqn(ref_val=ref, modelled_val=modelled, weight=scaled_wgt)
            for ref, modelled in lin_eqn_sides
            )

        # Now go on to the next property.
        continue

    return eqns


def _get_prop(props_dict, spec):
    """Gets the given property specification from the dictionary

    The specification can be a string or a pair of string and a callable.

    :param dict props_dict: The dictionary to retrieve the property.
    :param spec: The specification of the property, it can be a string, or a
        pair of a string and a callable. The string is going to be used as the
        key to retrieve the property, and the callable is going to be applied
        to the raw value of the property to get the final value.
    :returns: The value of the property.
    """

    if isinstance(spec, str):
        prop_tag = spec
        func = lambda x: x
    else:
        try:
            prop_tag = spec[0]
            func = spec[1]
            if not isinstance(func, abc.Callable):
                raise ValueError(
                    'Invalid callable on property {}!'.format(prop_tag)
                    )
        except IndexError:
            raise ValueError(
                'Invalid property specification {}'.format(spec)
                )

    return func(
        props_dict[prop_tag]
        )


def _compute_rms_deviation(ref_vals, modelled_vals):
    """Computes the RMS deviation of the reference and modelled values

    :param ref_vals: A linear list of reference values.
    :param modelled_vals: A linear list of modelled values.
    :returns: The RMS deviation of the modelled values from the reference
        values.
    :rtype: float
    """

    return math.sqrt(
        sum((i - j) ** 2 for i, j in zip(ref_vals, modelled_vals)) /
        len(ref_vals)
        )
