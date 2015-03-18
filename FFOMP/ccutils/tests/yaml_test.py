"""
Tests for the conversion to YAML facilities
===========================================

"""


import unittest
import os
import os.path

from cclib.parser import ccopen
from yaml import load

from .. import logfile2PESyaml


class YamlTest(unittest.TestCase):

    """
    Tests the conversion to YAML facility in the ccutils packages

    In the set-up function, the current working directory will be switched to
    the directory containing this test fixture, where there is a simple
    Gaussian output file for a single-point calculation. In the tear-down
    function, the old working directory will be restored.

    """

    def setUp(self):
        """Switches the CWD to the module path"""

        self._oldcwd = os.getcwd()
        os.chdir(
            os.path.dirname(__file__)
            )

    def read_test(self):
        """Tests the result of reading a Gaussian output"""

        symbs = []
        mols = []

        logfile2PESyaml(
            ccopen('sp.out'), 'sp.yml', symbs, mols,
            ('scfenergies', lambda x: x[-1] - 0.0)
            )

        with open('sp.yml', 'r') as res_file:
            res_dict = load(res_file)

        self.assertIn('atm_symbs', res_dict)
        self.assertEqual(res_dict['atm_symbs'], symbs)
        self.assertIn('mols', res_dict)
        self.assertEqual(res_dict['mols'], mols)

        self.assertIn('atm_coords', res_dict)
        coords = res_dict['atm_coords']
        self.assertEqual(len(coords), 10)


    def tearDown(self):
        """Switches back to the old working directory"""
        os.chdir(self._oldcwd)

