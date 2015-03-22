"""
Tests for the conversion to YAML facilities
===========================================

"""


import unittest
import os
import os.path
import gzip

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
        """Switches the CWD to the module path and unzip the output file"""

        # Switch the directory.
        self._oldcwd = os.getcwd()
        os.chdir(
            os.path.dirname(__file__)
            )

        # Unzip the output file.
        with gzip.open('sp.log.gz', 'rb') as compressed:
            with open('sp.log', 'wb') as uncompressed:
                uncompressed.write(compressed.read())

    def read_test(self):
        """Tests the result of reading a Gaussian output"""

        C = 'C'
        H1 = 'H1'
        H99 = 'H99'
        symbs = [
            C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, H1, H1, H1, H1, H1,
            H1, C, C, C, H1, H1, C, C, C, H1, H1, C, H1, C, H1, H99, H99
            ]

        mols = [
            list(range(0, 36)),
            list(range(36, 38))
            ]

        logfile2PESyaml(
            ccopen('sp.log'), 'sp.yml', symbs, mols,
            energy=('scfenergies', lambda x: float(x[-1]) - 1.0)
            )

        with open('sp.yml', 'r') as res_file:
            res_dict = load(res_file)

        self.assertIn('atm_symbs', res_dict)
        self.assertEqual(res_dict['atm_symbs'], symbs)
        self.assertIn('mols', res_dict)
        self.assertEqual(res_dict['mols'], mols)

        self.assertIn('atm_coords', res_dict)
        coords = res_dict['atm_coords']
        self.assertEqual(len(coords), 38)
        self.assertAlmostEqual(coords[37][2], 2.429172)

        self.assertIn('static_energy', res_dict)
        self.assertAlmostEqual(
            res_dict['static_energy'], -25125.94 - 1.0,
            places=2
            )

        self.assertIn('atm_forces', res_dict)
        forces = res_dict['atm_forces']
        self.assertEqual(len(forces), 38)
        self.assertEqual(forces[37][2], 0.003242665 * 51.42207)

        # Delete the temporary file.
        os.unlink('sp.yml')

    def tearDown(self):
        """Switches back to the old working directory"""

        # Remove the uncompressed file.
        os.unlink('sp.log')

        # Change back to the old working directory.
        os.chdir(self._oldcwd)
