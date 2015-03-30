"""
Tests for the conversion to YAML facilities
===========================================

"""


import unittest
import os
import os.path
import gzip

from yaml import load

from ..gau2yaml import gauout2PESyaml


class YamlTest(unittest.TestCase):

    """
    Tests the conversion to YAML facility in the ccutils packages

    In the set-up function, the current working directory will be switched to
    the directory containing this test fixture, where there is a simple
    Gaussian output file snippet for a single-point calculation. In the tear-
    down function, the old working directory will be restored.

    """

    def setUp(self):
        """Switches the CWD to the module path and unzip the output file"""

        # Switch the directory.
        self._oldcwd = os.getcwd()
        os.chdir(
            os.path.dirname(__file__)
            )

        # Unzip the output file.
        with gzip.open('he-test.out.gz', 'rb') as compressed:
            with open('he-test.out', 'wb') as uncompressed:
                uncompressed.write(compressed.read())

    def read_test(self):
        """Tests the result of reading a Gaussian output"""

        gauout2PESyaml(
            'he-test.out', 'he-test.yml',
            symbs=lambda idx, default: '{}{}'.format(default, idx),
            mols=[1, 1],
            )

        with open('he-test.yml', 'r') as res_file:
            res_dict = load(res_file)

        self.assertIn('atm_symbs', res_dict)
        self.assertEqual(
            res_dict['atm_symbs'], ['He0', 'He1']
            )
        self.assertIn('mols', res_dict)
        self.assertEqual(
            res_dict['mols'], [[0], [1]]
            )

        self.assertIn('atm_coords', res_dict)
        coords = res_dict['atm_coords']
        self.assertEqual(len(coords), 2)
        for i in coords[0]:
            self.assertAlmostEqual(i, 0.0)
        for i in coords[1]:
            self.assertAlmostEqual(i, 2.0)

        self.assertIn('static_energy', res_dict)
        self.assertAlmostEqual(
            res_dict['static_energy'], -158.5628, places=3
            )

        # Here we just test that
        #
        # 1. The three components of the forces should be equal.
        # 2. Newton's third law is satisfied.
        self.assertIn('atm_forces', res_dict)
        forces = res_dict['atm_forces']
        self.assertEqual(len(forces), 2)
        force_component = sum(forces[0]) / 3.0
        for i in forces[0]:
            self.assertAlmostEqual(i, force_component)
        for i in forces[1]:
            self.assertAlmostEqual(i, force_component * -1)

        # Delete the temporary file.
        os.unlink('he-test.yml')

    def tearDown(self):
        """Switches back to the old working directory"""

        # Remove the uncompressed file.
        os.unlink('he-test.out')

        # Change back to the old working directory.
        os.chdir(self._oldcwd)
