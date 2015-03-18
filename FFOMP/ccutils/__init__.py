"""
Utilities for computational chemistry
=====================================

In this package, some utilities are provided for doing fitting on computational
chemistry data, which include

.. autosummary::
    :toctree:

    logfile2yaml
    logfile2PESyaml

"""


from yaml import dump, YAMLError
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper


def logfile2yaml(logfile, yaml_file, props, add_info):
    """Dumps given properties in a cclib logfile object into a YAML file

    This function will attempt to get and convert the given properties in the
    cclib log file object and dumps them into a YAML file with the given name.

    :param logfile: The cclib logfile object, possibly from ccopen function or
        an individual parser. The ``parse`` method will be called on it to get
        the properties.
    :param str yaml_file: The name of the output YAML file.
    :param props: An interable for the desired properties. With each entry
        being a triple of the string for the name of the property in the ccData
        object, a callable for the processing of the data, which could be set
        to None for no processing, and the string tag for the new tag for the
        processed data in the result YAML.
    :param dict add_info: The additional information to be dumped into the
        YAML.
    :returns: None
    """

    # Parse the log file.
    data = logfile.parse()

    # The output dictionary to be dumped.
    out = {}
    for prop in props:

        try:
            raw_val = getattr(data, prop[0])
        except AttributeError:
            raise ValueError(
                'Invalid property {}'.format(prop[0])
                )

        if len(prop) > 1 and prop[1] is not None:
            proced_val = prop[1](raw_val)
        else:
            proced_val = raw_val

        if len(prop) > 2:
            tag = prop[2]
        else:
            tag = prop[0]
        out[tag] = proced_val

        continue

    out.update(add_info)

    try:
        with open(yaml_file, 'w') as out_file:
            dump(out, stream=out_file, Dumper=Dumper)
    except IOError:
        raise IOError(
            'Invalid output file {}'.format(yaml_file)
            )
    except YAMLError:
        raise ValueError(
            'Invalid data to be dumped by YAML:\n{!r}'.format(out)
            )

    return None


def logfile2PESyaml(logfile, yaml_file, symbs, mols, energy=None):
    """Dumps the relevant information in the log to a YAML file for PES scan

    For force-field parameters fitting, we normally just need the atomic
    coordinates, static energy, and the atomic forces. These data will be
    dumped under the labels ``atm_coords``, ``static_energy``,
    ``atomic_forces`` respectively. Also the atomic symbols and the division of
    the system into molecules needs to be given.

    :param logfile: The cclib log file object to extract the data.
    :param str yaml_file: The name of the YAML output file.
    :param symbs: The iterable of atomic symbols.
    :param mols: The nested list of atomic indices for the division of the
        system into molecules.
    :param energy: The property tag and processing function for energy, the SCF
        energy by default. Most of times, the reference value needs to be
        subtracted from the raw value of the energy.
    """

    energy = energy or ('scfenergies', lambda x: x[-1])

    return logfile2yaml(
        logfile, yaml_file,
        [
            ('atomcoords', lambda x: x[-1, :, :].tolist(), 'atm_coords'),
            energy + ('static_energy', ),
            ('grads', lambda x: x[-1, :, :].tolist(), 'atom_foces')
            ],
        {
            'atm_symbs': symbs,
            'mols': mols,
            }
        )
