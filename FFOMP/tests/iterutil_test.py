"""
Tests for the iteration utilities
=================================

"""


import unittest

import numpy as np
from numpy import random

from .._iterutil import flatten_zip, map_nested


class IterUtilTest(unittest.TestCase):

    """Tests the iteration utilities

    Two nested lists are used to test the functions on second-order tensors,
    with one being a random matrix of integers, and the other being its value
    plus one for each entry.

    """

    def setUp(self):
        """Sets up the test case"""

        self.mat = random.randint(
            low=0, high=100, size=(5, 5)
            )
        self.mat_plus_one = self.mat + 1

        self.mat_lst = self.mat.tolist()
        self.mat_plus_one_lst = self.mat_plus_one.tolist()

    def test_flatten_zip(self):
        """Tests the flatten zip"""

        comp = flatten_zip(self.mat_lst, self.mat_plus_one_lst)
        for i, j in comp:
            self.assertEqual(i + 1, j)
            continue

    def test_map_nested(self):
        """Tests the nested map"""

        ref = np.array(
            map_nested(lambda x: x + 1, self.mat_lst)
            )
        self.assertTrue(
            np.array_equal(self.mat_plus_one, ref)
            )
