"""
Models for two-body pairwise interactions
=========================================

.. currentmodule:: FFOMP.mmmodels.twobody

In this module, models for pairwise atomic interactions has been defined. And
they are all based on an abstract base class where most of the aspects of two-
body interactions are defined. The models includes,

.. autosummary::
    :toctree:

    Morse
    CubicSpline
    LeonardJones
    _TwoBodyI

"""


import abc
import itertools
import functools

from sympy import Float, Symbol, exp
from sympy.functions.special.bsplines import bspline_basis_set
import numpy as np
from numpy import linalg

from ..model import Model
from ..fitjob import ModelParam


#
# The abstract base class
# -----------------------
#


class _TwoBodyI(Model):

    """
    The abstract base class for two body interactions

    The subclasses of this class are required to define their own parameters
    and presentation methods. One more property ``energy_expr`` needs to be
    defined which gives a tuple whose first field is the expression of the
    interaction energy in terms of the models parameters and a distance symbol
    on the second field. Also the atomic symbols for the atomic types that the
    interaction applies to should be in the property ``atm_types``. Then the
    actual computation of static energy and atomic forces will be automatically
    carried out by the :py:meth:`__call__` method of this base class, based on
    the given energy profile expression. Note that the ``mols`` property in the
    data point are going to be used so that only interactions between atoms of
    different molecules will be included.

    """

    def __init__(self, force_match=True):
        """Initializes a two-body interaction

        Currently it just sets if the forces are going to be modelled as well.

        :param bool force_match: If the forces are going to be matched.
        """

        self._force_match = force_match

    @abc.abstractproperty
    def energy_expr(self):
        """The expression of the interaction energy

        It should be a tuple with the first field being the sympy expression of
        energy and the second field being the symbol that is used for the
        atomic distance in the expression.
        """

        pass

    @abc.abstractproperty
    def atm_types(self):
        """The types of the atoms that this interaction describes

        It needs to be a tuple of two strings, giving the symbols for the type
        of atoms whose interaction this model describes.
        """

        pass

    def __call__(self, data_pnt):
        """Calculates the static energy and atomic forces for the data point

        The staic energy and atomic forces will be computed.
        """

        # Before anything, make sure that we are operating on a valid data
        # point for potential energy surface scan fitting.
        try:
            symbs = data_pnt['atm_symbs']
            coords = data_pnt['atm_coords']
        except KeyError as exc:
            raise ValueError(
                'Property {} is required for two-body interaction '
                'models'.format(exc.args[0])
                )
        mols = data_pnt.get(
            'mols', [[i, ] for i, _ in enumerate(symbs)]
            )

        # First, get all the interacting pairs.
        pairs = _get_interacting_pairs(symbs, mols, self.atm_types)

        # Allocate the results and gradually add to them for each interacting
        # pair.
        energy = Float(0)
        if self._force_match:
            forces = [
                list(itertools.repeat(Float(0), 3))
                for _ in symbs
                ]

        # Lambdify the expression for energy and forces.
        energy_expr, dist_symb = self.energy_expr

        def energy_func(dist):
            """The energy function"""
            return energy_expr.subs(dist_symb, dist)

        if self._force_match:

            force_mag_expr = energy_expr.diff(dist_symb) * -1

            def force_mag_func(dist):
                """The force magnitude function"""
                return force_mag_expr.subs(dist_symb, dist)

        # Iterate over all the interacting pairs.
        for atm1, atm2 in pairs:

            # Calculate the distance and normalized vector from atom 1 to atom
            # 2.
            dist, vec = _get_dist_vec(coords[atm1], coords[atm2])

            # Test the cut-off.
            if hasattr(self, 'cut_off') and dist > self.cut_off:
                continue

            # Add the contribution to energy.
            energy += energy_func(dist)

            if self._force_match:
                # Compute the magnitude of the force.
                force_mag = force_mag_func(dist)

                # Add the force for the two atoms.
                for i in [atm1, atm2]:
                    for j, k in enumerate(vec):
                        forces[i][j] += k * force_mag
                        # Continue to the next axis
                        continue
                    # Flip the vector.
                    vec *= -1
                    # Continue to the next atom.
                    continue

            # Continue to the next atomic pair.
            continue

        # Return the result.
        return {
            'static_energy': energy,
            'atm_forces': forces,
            }


def _get_interacting_pairs(symbs, mols, symbs4inter):
    """Gets the list of interacting pairs on the system

    The pairs of indices of atomic in the symbol list will be returned. The
    interacting pairs must have the symbols required by the interaction and
    belongs to different molecules.

    :param symbs: The sequence of atomic symbols in the system.
    :param mols: A sequence of sequences of atomic indices, with each
        subsequence gives the indices of atomic of one molecule.
    :param symbs4inter: The pair of atomic symbols for the interaction.
    :returns: The list of atomic indices pairs of the interacting pairs of
        atoms for this interaction.
    :rtype: list
    """

    symbs4inter = tuple(sorted(symbs4inter))

    pairs = []

    for mol1, mol2 in itertools.combinations(mols, 2):
        for atms in itertools.product(mol1, mol2):
            if tuple(sorted(symbs[i] for i in atms)) == symbs4inter:
                pairs.append(atms)
            else:
                continue

    return pairs


def _get_dist_vec(pnt1, pnt2):
    """Gets the distance and normalized vector between two points

    The normalized vector will be an numpy vector from point one to point two.
    The two points do not need to be numpy arrays before hand.

    :param pnt1: The first point, as a iterable of coordinate values.
    :param pnt2: The second point.
    :returns: The Euclidean distance between the two points and the normalized
        vector from point one to point two.
    :rtype: tuple:
    """

    diff = np.array(
        [j - i for i, j in zip(pnt1, pnt2)],
        dtype=np.float
        )

    dist = linalg.norm(diff)

    return dist, diff / dist


#
# Potentials
# ----------
#


class Morse(_TwoBodyI):

    """The Morse interaction potential

    In this three parameter model, the interaction potential reads

    .. math::
        D_e \\left((1 - e^{-a(r - r_e)})^2 - 1\\right)

    """

    def __init__(self, atm_symbs, params, force_match=True):
        """Initializes a Morse interaction model

        :param atm_symbs: A pair of strings for the atomic symbols of the atoms
            whose interaction this model describes.
        :param params: The triple of initial guess, lower bound, and upper
            bound for the parameters, in the order of :math:`D_e`, :math:`a`,
            :math:`r_0`.
        """

        super().__init__(force_match=force_match)

        self._atm_types = tuple(str(i) for i in atm_symbs)
        if len(self._atm_types) != 2:
            raise ValueError(
                'Invalid number of atomic symbols!'
                )

        param_name_bases = [
            'DE', 'a', 'r0'
            ]
        self._model_params = []
        for name_base, spec in zip(param_name_bases, params):
            try:
                self._model_params.append(ModelParam(
                    symb=Symbol('_'.join((name_base, ) + self._atm_types)),
                    lower=spec[1], upper=spec[2],
                    init_guess=spec[0],
                    ))
            except IndexError:
                raise ValueError(
                    'Invalid parameter specification {} for {}'.format(
                        spec, name_base
                        )
                    )

    @property
    def atm_types(self):
        """The atomic types of interactions"""
        return self._atm_types

    @property
    def energy_expr(self):
        """The energy expression"""

        r = Symbol('r')
        de, a, r0 = [i.symb for i in self._model_params]

        return (
            de * ((exp(a * (r0 - r)) - 1) ** 2 - 1),
            r
            )

    @property
    def model_params(self):
        """The model parameters"""
        return self._model_params

    def present(self, param_vals, output):
        """Presents the results

        The results and a restart literal for the current object will be
        written.

        :param param_vals: A sequence for the values of the parameters.
        :param output: The output stream.
        """

        prt = functools.partial(print, file=output)

        prt('\n\n')
        prt('Morse parameters for {0[0]} and {0[1]}: De  a  r0'.format(
            self._atm_types
            ))
        prt('{0[0]!r}  {0[1]!r}  {0[2]!r}'.format(param_vals))
        prt('Restart literal:')
        prt('Morse({atm_types!r}, {params!r})'.format(
            atm_types=self._atm_types,
            params=[
                (j, i.lower, i.upper)
                for i, j in zip(self.model_params, param_vals)
                ]
            ))
        prt('\n\n')

        return None


class CubicSpline(_TwoBodyI):

    """
    Cubic spline two-body interaction model

    In this cubic spline model, the interaction energy is a linear
    superposition of cubic basis splines.
    """

    def __init__(self, atm_symbs, knots, force_match=True):
        """Initializes a cubic spline interaction model

        :param atm_symbs: A pair of strings for the atomic symbols of the atoms
            whose interaction this model describes.
        :param knots params: The knots in the cubic spline, given as a sequence
            of numbers for the locations. The last location will be taken as
            the cut-off for the interaction.
        """

        super().__init__(force_match=force_match)

        self._atm_types = tuple(str(i) for i in atm_symbs)
        if len(self._atm_types) != 2:
            raise ValueError(
                'Invalid number of atomic symbols!'
                )

        param_name_base = 'alpha'
        self._dist = Symbol('r')
        self._model_params = []
        self._basis = []
        self._knots = knots
        self.cut_off = max(knots)

        for idx, basis in enumerate(bspline_basis_set(3, knots, self._dist)):
            self._model_params.append(ModelParam(
                symb=Symbol(''.join([
                    param_name_base, self._atm_types[0], self.atm_types[1],
                    str(idx)
                    ])),
                lower=None, upper=None, init_guess=None
                ))
            self._basis.append(basis)
            continue

        self._energy_expr = sum(
            i * j.symb for i, j in zip(self._basis, self._model_params)
            )

    @property
    def atm_types(self):
        """The atomic types of interactions"""
        return self._atm_types

    @property
    def energy_expr(self):
        """The energy expression"""

        return (
            self._energy_expr, self._dist
            )

    @property
    def model_params(self):
        """The model parameters"""
        return self._model_params

    def present(self, param_vals, output):
        """Presents the results

        Very different from other models, this model will present itself as a
        list of energies for the knots points.

        :param param_vals: A sequence for the values of the parameters.
        :param output: The output stream.
        :returns: The list of the energy values at the knots points will be
            returned.
        :rtype: list
        """

        prt = functools.partial(print, file=output)

        prt('\n\n')
        prt('Cubic spline fitting results for {0[0]} and {0[1]}\n'.format(
            self._atm_types
            ))

        vals = []
        subs_dict = {
            i.symb: j
            for i, j in zip(self._model_params, param_vals)
            }

        for i in self._knots:
            subs_dict[self._dist] = i
            vals.append(
                float(self._energy_expr.subs(subs_dict).evalf())
                )
            prt('   {}     {}'.format(i, vals[-1]))
            continue
        prt('\n\n')

        return None
