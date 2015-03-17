"""
Tests the solvers with a simple linear problem
==============================================

"""


import unittest
import random
import itertools
import operator

from sympy import Symbol

from ..model import Model
from ..fitjob import ModelParam, FitJob, add_elementwise
from ..solvers.linear import numpy_lstsq, r_lm
from ..solvers.scipyoptimize import so_minimize, so_leastsq


class LinearModel(Model):

    """Simple linear model to test the solvers

    It just has got two model parameters,  which are the coefficients ahead of
    the input field one and input field two for the one of the fields of the
    output field. The initial guess are all zero. And they are bounded from
    minus to plus ten.
    """

    def __init__(self, output_dim):
        """Initialize the model

        For any given output dimension, the model will only model the given
        field of the output property as a linear combination of the two input
        properties. So two such models are needed to model the toy problem.

        The model parameters will be stored in attribute ``_model_params``.
        """

        self._output_dim = output_dim

        symb_names = [i + str(output_dim) for i in ['a', 'b']]
        self._model_params = [
            ModelParam(
                symb=Symbol(i),
                upper=10, lower=-10, init_guess=0.0
                )
            for i in symb_names
            ]

    @property
    def model_params(self):
        """Gets the model parameters"""
        return self._model_params

    def __call__(self, data_pnt):
        """Make modelling"""

        res = [0, 0]
        res[self._output_dim] = (
            data_pnt['inp1'] * self._model_params[0].symb +
            data_pnt['inp2'] * self._model_params[1].symb
            )

        return {
            'outp': res
            }

    def present(self, param_vals, output):
        """Present the results"""

        print(
            ' {} = {}; {} = {}.'.format(
                self._model_params[0].symb, param_vals[0],
                self._model_params[1].symb, param_vals[1],
                ),
            file=output
            )

        return None


class SolversSimpleTest(unittest.TestCase):

    """
    A simple test for the solvers on a simple linear problem

    Raw data points are going to be generated with properties ``inp1`` and
    ``inp2``, and output property ``outp``, which is a list of the sum and
    difference of the two input fields.

    """

    def setUp(self):
        """Sets up the simple test case

        Raw data and models are all going to be added to the test case. The
        solvers could directly use them.
        """

        # Raw data points, just sets of two random numbers.
        self.raw_data = []
        self.n_data = 50
        for i in range(0, self.n_data):
            inp1 = random.random()
            inp2 = random.random()
            self.raw_data.append({
                'inp1': inp1, 'inp2': inp2,
                'outp': [inp1 + inp2, inp1 - inp2]
                })
            continue

        # Add the models.
        self.models = [
            LinearModel(i) for i in range(0, 2)
            ]

        # Add the property merger
        self.prop_merger = {
            'outp': add_elementwise,
            }

    def _check_result(self, res):
        """Checks the correctness of the results"""

        symbs = [
            j.symb
            for j in itertools.chain.from_iterable(
                i.model_params for i in self.models
                )
            ]
        for i in symbs[0:3]:
            self.assertAlmostEqual(res[i], 1.0)
            continue
        self.assertAlmostEqual(res[symbs[3]], -1.0)

        return None

    def _check_compare_res(self, job):
        """Checks the result comparison"""

        # Get the two input random vectors.
        inp1 = [i['inp1'] for i in self.raw_data]
        inp2 = [i['inp2'] for i in self.raw_data]

        # First check the default coordinate parameter.
        coord_param, ref_val, modelled_val = job.compare_prop(
            None, ('outp', operator.itemgetter(0))
            )
        self.assertEqual(coord_param, list(range(0, self.n_data)))
        for i, j in zip(ref_val, modelled_val):
            self.assertAlmostEqual(i, j)
            continue
        for i, j, k in zip(ref_val, inp1, inp2):
            self.assertAlmostEqual(i, j + k)
            continue

        # Check the given coordinate parameter
        coord_param, _, _ = job.compare_prop(
            'inp1', ('outp', operator.itemgetter(1))
            )
        for i, j in zip(inp1, coord_param):
            self.assertAlmostEqual(i, j)
            continue

    def test_numpy_linear(self):
        """Tests the plain numpy solver"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models, numpy_lstsq(),
            prop_merger=self.prop_merger
            )
        self._check_result(res)
        self._check_compare_res(job)

    def test_r_lm(self):
        """Tests the R lm solver"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models, r_lm(),
            prop_merger=self.prop_merger
            )
        self._check_result(res)

    def test_minimize_no_bounds(self):
        """Tests the solver based on scipy.optimize.minimize, no bounds"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models,
            so_minimize(use_bounds=False, method='Newton-CG', tol=1.0E-10),
            prop_merger=self.prop_merger
            )
        self._check_result(res)

    def test_minimize_no_hess(self):
        """Tests the solver based on scipy.optimize.minimize, no Hessian"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models,
            so_minimize(use_hess=False, tol=1.0E-15),
            prop_merger=self.prop_merger
            )
        self._check_result(res)

    def test_leastsq(self):
        """Tests the solver based on scipy.optimize.leastsq"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models,
            so_leastsq(xtol=1.0E-10),
            prop_merger=self.prop_merger
            )
        self._check_result(res)
