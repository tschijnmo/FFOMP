"""
Tests of the Morse interaction potential
========================================

"""


import unittest
import math

import numpy as np
from numpy.linalg import norm

from ..twobody import Morse


#
# The hard-coded Python code for Morse potential
# ----------------------------------------------
#


def morse_energy(dist, params):
    """The Morse interaction energy

    :param dist: The distance between the two interacting atoms.
    :param params: The Morse parameters.
    :returns: The interaction energy.
    """

    de, a, r0 = params
    return de * ((
            math.exp(a * (r0 - dist)) - 1
            ) ** 2 - 1.0)


def morse_force(dist, params):
    """The Morse interaction force

    This function is already the negation of the derivative of the Morse energy
    with respect to the distance.
    """

    de, a, r0 = params
    return (
        -2 * a * de * math.exp(-a * (dist - r0)) *
        (1 - math.exp(-a * (dist - r0)))
        )


#
# The test case
# -------------
#


class MorseTest(unittest.TestCase):

    """
    Tests of the Morse potential

    This test is also to test the basic facilities in the two-body interaction
    abstract base class.

    In this test case, one hydrogen molecule is going to be put on the x axis
    and one helium atom is going to be put on the origin, with another neon
    atom in a random position. The Morse model for the interaction between
    hydrogen atom and helium atom should give the same energy and force as the
    Python code defined in this module.
    """

    def setUp(self):
        """Sets up the test case"""

        # The location and bound length for the hydrogen molecule.
        self.h_loc = 5.0
        self.h_bl = 0.5

        self.data_pnt = {
            'atm_symbs': ['He', 'H', 'H', 'Ne'],
            'mols': [[0, ], [1, 2], [3, ]],
            'atm_coords': [
                [0.0, 0.0, 0.0],
                [self.h_loc + self.h_bl, 0.0, 0.0],
                [self.h_loc - self.h_bl, 0.0, 0.0],
                [3.423, 4.0, 1.34],
                ],
            }

        self.morse_params = (2.0, 1.0, 5.0)
        self.model = Morse(
            ('He', 'H'), [(i, None, None) for i in self.morse_params]
            )
        self.subs = dict(zip(
            (i.symb for i in self.model.model_params),
            self.morse_params
            ))

    def test_energy(self):
        """Tests the Morse interaction energy"""

        # First apply the model to the data point.
        res = self.model(self.data_pnt)
        self.assertIn('static_energy', res)

        energy = res['static_energy'].evalf(subs=self.subs)
        correct_energy = (
            morse_energy(self.h_loc + self.h_bl, self.morse_params) +
            morse_energy(self.h_loc - self.h_bl, self.morse_params)
            )
        self.assertAlmostEqual(energy, correct_energy)

    def test_force(self):
        """Tests the Morse interacting force"""

        # First apply the model to the data point.
        res = self.model(self.data_pnt)
        self.assertIn('atm_forces', res)

        forces = [
            np.array([j.evalf(subs=self.subs) for j in i], dtype=np.float)
            for i in res['atm_forces']
            ]

        self.assertEqual(len(forces), 4)

        force_1 = np.array([
            morse_force(self.h_loc + self.h_bl, self.morse_params), 0.0, 0.0
            ])
        self.assertAlmostEqual(
            norm(force_1 - forces[1]), 0.0
            )

        force_2 = np.array([
            morse_force(self.h_loc - self.h_bl, self.morse_params), 0.0, 0.0
            ])
        self.assertAlmostEqual(
            norm(force_2 - forces[2]), 0.0
            )

        # Action equals reaction.
        self.assertAlmostEqual(
            norm(-force_1 - force_2 - forces[0]), 0.0
            )

        self.assertAlmostEqual(
            norm(forces[3]), 0.0
            )



