"""
Models for molecular mechanics
==============================

In this package, models are provided to model some molecular and material
properties based on some force fields.

Most often force fields are going to model the energy and atomic forces based
on the atomic coordinates and their symbols (types). For these situations, the
convention is that the atomic symbol and coordinates needs to be in property
``atm_symbs`` and ``atm_coords``, both of which should be a list with one entry
of each of the atoms. Then based on these information, possibly some other bond
connectivity information, the interaction energy of the atoms will be modelled
and put into a property with tag ``static_energy``, also the forces will also
be modelled and put in the property ``atm_forces``. This convention is going to
be followed by all the models in this packages.

.. autosummary::
    :toctree:
    :template: module.rst

    twobody

"""
