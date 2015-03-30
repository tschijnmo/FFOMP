"""
Output conversion for Gaussian
==============================

This module contains conversion utilities that is solely written for the
Gaussian computational chemistry program.

.. autosummary::
    :toctree:

    gauout2PESyaml

"""


import collections
import re
from collections import abc
import itertools

import numpy as np

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
from yaml import dump, YAMLError


#
# The drive function
# ------------------
#


def gauout2PESyaml(gauout_name, yaml_name,
                   energy_patt=r'^ SCF Done[^=]+=(?P<energy>[^A]+)A\.U',
                   ref_energy=0.0, symbs=None, mols=None):
    """Converts a Gaussian output file to a PES YAML file

    The atomic coordinates will be stored in the field ``atm_coords`` in input
    orientation in units of Angstrom. The SCF energy will be stored as
    ``static_energy`` in units of eV. The forces will be stored in
    ``atm_foces`` in the unit of eV/Angstrom.

    The atomic symbols and molecules will also be stored in ``atm_symbs`` and
    ``mols`` according to user input.

    :param str gauout_name: The name of the Gaussian output file.
    :param str yaml_name: The name of the YAML file to be written.
    :param str energy_patt: The pattern that can be used to grab the raw energy
        in Hartree. The energy needs to be in the named group ``energy`` and
        the last line matching the pattern with search will be used. Default to
        the SCF energy.
    :param float ref_energy: The reference energy to be subtracted from the raw
        energy, in Hartree.
    :param symbs: The symbols for the atoms in the output. By default the
        element symbol for the atomic numbers will be used. Or it can be given
        as a callable which will be called with the atomic index number and the
        default symbol to return the actual symbol of the atoms. An iterable
        can be given directly as well.
    :param mols: An iterable for the atomic indices of the molecules in the
        system. Elements in the iterable can be another iterable to give the
        actual indices of the atoms, or an integral number to show that the
        next n atoms will be a molecule. By default there is going to be just
        one molecule.
    :raises ValueError: if the input has got problems.
    :raises IOError: if something is wrong with the files.
    :returns: 0 for success.
    """

    # Parse the Gaussian output.
    parse_res = _parse_gauout(gauout_name, energy_patt)

    # The result dictionary.
    res = {}

    # The coordinates.
    res['atm_coords'] = parse_res.atm_coords.tolist()
    # The energy.
    res['static_energy'] = (
        parse_res.static_energy - ref_energy
        ) * _HARTREE2EV
    # The forces.
    res['atm_forces'] = (
        parse_res.atm_forces * _HARTREE_P_BOHR2EV_P_ANGS
        ).tolist()

    atm_numbs = parse_res.atm_numbs
    # The symbols.
    res['atm_symbs'] = _gen_symbs(atm_numbs, symbs)
    # The molecules.
    res['mols'] = _gen_mols(atm_numbs, mols)

    # Dump to the YAML file.
    _dump2yaml(yaml_name, res)

    return 0


#
# Some unit conversion constants
# ------------------------------
#


_HARTREE2EV = 27.21139
_HARTREE_P_BOHR2EV_P_ANGS = 51.42207


#
# Gaussian output parsing
# -----------------------
#


ParseRes = collections.namedtuple(
    'ParseRes', [
        'atm_coords',
        'static_energy',
        'atm_forces',
        'atm_numbs',
        ]
    )


def _parse_gauout(gauout_name, energy_patt):
    """Parses the given Gaussian output file

    The results will be put in a named tuple. All units are *not* converted.
    And tensor properties like coordinates and forces will be in numpy arrays.

    :param str gauout_name: The name of the Gaussian output file to parse.
    :param str energy_patt: The energy pattern to grab the energy.
    :returns: The parse result.
    """

    # Open and read the file.
    try:
        with open(gauout_name, 'r') as gauout:
            lines = gauout.readlines()
    except IOError:
        raise

    # Get the energy, the easiest one.
    compiled_energy_patt = re.compile(energy_patt)
    static_energy = None
    for line in lines:
        res = compiled_energy_patt.search(line)
        if res is None:
            continue
        else:
            static_energy = float(res.group('energy'))
        continue
    if static_energy is None:
        raise ValueError(
            'Energy failed to be read from {}'.format(gauout_name)
            )

    # Get the coordinates and the atomic numbers.
    coords_lines = _get_lines_under_title(
        lines, r'^ +Input orientation: *$', r'^ *\d'
        )
    atm_numbs = []
    atm_coords = []
    for line in coords_lines:
        fields = line.split()
        atm_numbs.append(
            int(fields[1])
            )
        atm_coords.append(
            [float(i) for i in fields[3:6]]
            )
        continue
    atm_coords = np.array(atm_coords)

    # Get the forces.
    forces_lines = _get_lines_under_title(
        lines, r'^ +\*+ +Axes restored to original set +\*+ *$', r'^ *\d'
        )
    atm_forces = []
    for line in forces_lines:
        fields = line.split()
        atm_forces.append(
            [float(i) for i in fields[2:5]]
            )
        continue
    atm_forces = np.array(atm_forces)

    return ParseRes(
        atm_coords=atm_coords, static_energy=static_energy,
        atm_forces=atm_forces, atm_numbs=atm_numbs,
        )


def _get_lines_under_title(lines, title_patt, content_patt):
    """Gets the lines under a title

    If multiple titles are found, only the lines in the last section will be
    returned.

    :param lines: A sequence of lines.
    :param title_patt: The pattern for the title.
    :param content_patt: The pattern for the content lines.
    :raises ValueError: If the title cannot be found.
    :returns: The content lines following the title.
    """

    # Compile the given patterns
    compiled_title_patt = re.compile(title_patt)
    compiled_content_patt = re.compile(content_patt)

    # Find the location of the title.
    title_loc = None
    for idx, line in enumerate(lines):
        if compiled_title_patt.search(line) is not None:
            title_loc = idx
            continue
        else:
            continue
    if title_loc is None:
        raise ValueError(
            'The given title {} failed to be found'.format(title_patt)
            )

    # Gather the content lines following the title.
    content_lines = []
    started = False
    for line in lines[title_loc:]:

        if compiled_content_patt.search(line) is None:
            if started:
                break
            else:
                continue
        else:
            content_lines.append(line)
            if not started:
                started = True

    return content_lines


#
# Symbols and molecules generation
# --------------------------------
#


def _gen_symbs(atm_numbs, symbs):
    """Generates the atomic symbols

    By default, the element symbols will be used. If iterable is given its
    content will be directly used. If callable is given, it will be called with
    atomic index and default symbol to get the actual symbol.
    """

    if isinstance(symbs, abc.Iterable):

        symbs = list(symbs)
        if len(symbs) != len(atm_numbs):
            raise ValueError(
                'The given symbols does not match the number of atoms!'
                )

    else:

        default_symbs = [
            _ELEMENT_SYMBS[i] for i in atm_numbs
            ]

        if symbs is None:
            return default_symbs
        else:
            return [
                symbs(idx, default_symb)
                for idx, default_symb in enumerate(default_symbs)
                ]


def _gen_mols(atm_numbs, mols):
    """Generates the nested molecules list"""

    if mols is None:
        return [i for i, _ in enumerate(atm_numbs)]
    else:
        ret_val = []

    # Get the molecules list.
    curr_atm = 0
    for i in mols:

        if isinstance(i, int):
            ret_val.append(
                list(range(curr_atm, curr_atm + i))
                )
            curr_atm += 1
        else:
            ret_val.append(
                list(i)
                )
            curr_atm = max(i)

        continue

    # Check the correctness.
    for i, j in itertools.zip_longest(
            range(0, len(atm_numbs)),
            sorted(itertools.chain.from_iterable(ret_val))
            ):
        if i != j:
            raise ValueError(
                'Incorrect molecule specification, atom {} not correctly '
                'given!'.format(i)
                )
        continue

    return ret_val


_ELEMENT_SYMBS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    }


#
# Output generation
# -----------------
#


def _dump2yaml(yaml_name, content):
    """Dumps the content dictionary into a YAML file with the given name"""

    try:
        with open(yaml_name, 'w') as yaml_file:
            dump(content, stream=yaml_file, Dumper=Dumper)
    except IOError:
        raise IOError(
            'Invalid output file {}'.format(yaml_name)
            )
    except YAMLError:
        raise ValueError(
            'Invalid data to be dumped by YAML:\n{!r}'.format(content)
            )
