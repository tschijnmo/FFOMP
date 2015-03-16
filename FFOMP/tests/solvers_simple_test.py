"""
Tests the solvers with a simple linear problem
==============================================

"""


import unittest
import random
import itertools

from sympy import Symbol

from ..model import Model
from ..fitjob import ModelParam, FitJob
from .._iterutil import flatten_zip
from ..solvers.linear import numpy_lstsq, r_lm


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
        for i in range(0, 100):
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
            'outp': lambda res1, res2: [i + j for i, j in zip(res1, res2)],
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

    def test_numpy_linear(self):
        """Tests the plain numpy solver"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models, numpy_lstsq(),
            prop_merger=self.prop_merger
            )
        self._check_result(res)

    def test_r_lm(self):
        """Tests the R lm solver"""

        job = FitJob()
        res = job.fitting_driver(
            self.raw_data, self.models, r_lm(),
            prop_merger=self.prop_merger
            )
        self._check_result(res)
